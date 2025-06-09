import yaml
import argparse
import torch
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
from models.unet_cond_base import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils.text_utils import get_tokenizer_and_model, get_text_representation
from utils.config_utils import get_config_value
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_models(config):
    """Load trained models"""
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
    
    # Load VQVAE
    vqvae = VQVAE(im_channels=dataset_config['im_channels'],
                  model_config=autoencoder_model_config).to(device)
    
    vqvae_path = os.path.join(train_config['task_name'], train_config['vqvae_autoencoder_ckpt_name'])
    if os.path.exists(vqvae_path):
        vqvae.load_state_dict(torch.load(vqvae_path, map_location=device))
        print(f"Loaded VQVAE from {vqvae_path}")
    else:
        raise FileNotFoundError(f"VQVAE checkpoint not found at {vqvae_path}")
    
    vqvae.eval()
    
    # Load Diffusion Model
    ldm = Unet(im_channels=autoencoder_model_config['z_channels'],
               model_config=diffusion_model_config).to(device)
    
    ldm_path = os.path.join(train_config['task_name'], train_config['ldm_ckpt_name'])
    if os.path.exists(ldm_path):
        ldm.load_state_dict(torch.load(ldm_path, map_location=device))
        print(f"Loaded LDM from {ldm_path}")
    else:
        raise FileNotFoundError(f"LDM checkpoint not found at {ldm_path}")
    
    ldm.eval()
    
    # Load text models
    condition_config = diffusion_model_config['condition_config']
    text_tokenizer, text_model = get_tokenizer_and_model(
        condition_config['text_condition_config']['text_embed_model'], 
        device=device)
    
    return vqvae, ldm, text_tokenizer, text_model


def preprocess_sar_image(image_path, im_size=256):
    """Preprocess SAR image for conditioning"""
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize(im_size),
        transforms.CenterCrop(im_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    
    return transform(image).unsqueeze(0)  # Add batch dimension


def sample_with_conditioning(ldm, scheduler, sar_image, text_prompt, text_tokenizer, text_model, 
                           diffusion_config, autoencoder_config, condition_config, guidance_scale=7.5):
    """
    Sample optical image conditioned on SAR image and text prompt
    """
    im_size = 256 // 2 ** sum(autoencoder_config['down_sample'])
    
    # Prepare text conditioning
    with torch.no_grad():
        text_embed = get_text_representation([text_prompt], text_tokenizer, text_model, device)
        empty_text_embed = get_text_representation([''], text_tokenizer, text_model, device)
    
    # Initialize random noise
    xt = torch.randn((1, autoencoder_config['z_channels'], im_size, im_size)).to(device)
    
    # Prepare conditioning inputs
    cond_input = {
        'text': text_embed,
        'image': sar_image.to(device)
    }
    
    # Unconditional inputs for classifier-free guidance
    uncond_input = {
        'text': empty_text_embed,
        'image': torch.zeros_like(sar_image).to(device)
    }
      # Sampling loop
    for i in tqdm(reversed(range(diffusion_config['num_timesteps'])), desc="Sampling"):
        t = torch.full((1,), i, dtype=torch.long).to(device)
        
        with torch.no_grad():
            # Conditional prediction
            noise_pred_cond = ldm(xt, t, cond_input=cond_input)
            
            # Unconditional prediction for classifier-free guidance
            noise_pred_uncond = ldm(xt, t, cond_input=uncond_input)
            
            # Apply classifier-free guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            # Denoise step - use correct method name and handle return values
            # Convert tensor timestep to scalar for scheduler
            xt, x0 = scheduler.sample_prev_timestep(xt, noise_pred, i)
    
    return xt


def colorize_sar_image(sar_image_path, text_prompt, config, output_path=None, guidance_scale=7.5):
    """
    Main function to colorize a SAR image using text guidance
    """
    # Load models
    vqvae, ldm, text_tokenizer, text_model = load_models(config)
    
    # Load and preprocess SAR image
    sar_image = preprocess_sar_image(sar_image_path, config['dataset_params']['im_size'])
    
    # Create scheduler
    diffusion_config = config['diffusion_params']
    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end']
    )
    
    # Sample optical image
    print(f"Generating optical image for: {text_prompt}")
    with torch.no_grad():
        latent = sample_with_conditioning(
            ldm, scheduler, sar_image, text_prompt, text_tokenizer, text_model,
            diffusion_config, config['autoencoder_params'], 
            config['ldm_params']['condition_config'], guidance_scale
        )
        
        # Decode latent to image
        optical_image = vqvae.decode(latent)
    
    # Post-process image
    optical_image = (optical_image + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
    optical_image = torch.clamp(optical_image, 0, 1)
    
    # Save or return image
    if output_path:
        save_image(optical_image, output_path)
        print(f"Saved generated image to: {output_path}")
    
    return optical_image


def batch_inference(sar_folder, output_folder, config, default_region="unknown", default_season="unknown", guidance_scale=7.5):
    """
    Perform batch inference on a folder of SAR images
    """
    import glob
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all SAR images
    sar_files = glob.glob(os.path.join(sar_folder, "*.png")) + \
                glob.glob(os.path.join(sar_folder, "*.jpg")) + \
                glob.glob(os.path.join(sar_folder, "*.jpeg"))
    
    print(f"Found {len(sar_files)} SAR images for batch inference")
    
    for sar_file in tqdm(sar_files, desc="Processing SAR images"):
        # Generate text prompt
        text_prompt = f"Colorise image, Region: {default_region}, Season: {default_season}"
        
        # Generate output filename
        filename = os.path.splitext(os.path.basename(sar_file))[0]
        output_path = os.path.join(output_folder, f"{filename}_colorized.png")
        
        try:
            # Generate optical image
            colorize_sar_image(sar_file, text_prompt, config, output_path, guidance_scale)
        except Exception as e:
            print(f"Error processing {sar_file}: {str(e)}")
            continue


def interactive_inference(config):
    """
    Interactive inference mode
    """
    print("=== Interactive SAR Image Colorization ===")
    print("Enter 'quit' to exit")
    
    while True:
        # Get input from user
        sar_path = input("\nEnter path to SAR image: ").strip()
        if sar_path.lower() == 'quit':
            break
        
        if not os.path.exists(sar_path):
            print("File not found. Please try again.")
            continue
        
        region = input("Enter region (e.g., urban, forest, water): ").strip()
        if not region:
            region = "unknown"
        
        season = input("Enter season (spring, summer, autumn, winter): ").strip()
        if not season:
            season = "unknown"
        
        guidance_scale = input("Enter guidance scale (default 7.5): ").strip()
        try:
            guidance_scale = float(guidance_scale) if guidance_scale else 7.5
        except ValueError:
            guidance_scale = 7.5
        
        # Generate text prompt
        text_prompt = f"Colorise image, Region: {region}, Season: {season}"
        
        # Generate output path
        output_dir = "inference_results"
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.splitext(os.path.basename(sar_path))[0]
        output_path = os.path.join(output_dir, f"{filename}_colorized.png")
        
        try:
            # Generate optical image
            print(f"Generating image with prompt: '{text_prompt}'")
            colorize_sar_image(sar_path, text_prompt, config, output_path, guidance_scale)
            print(f"Generated image saved to: {output_path}")
        except Exception as e:
            print(f"Error during inference: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='SAR to Optical Image Translation Inference')
    parser.add_argument('--config', default='config/sar_config.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--mode', choices=['single', 'batch', 'interactive'], 
                       default='interactive', help='Inference mode')
    parser.add_argument('--sar_image', help='Path to SAR image (for single mode)')
    parser.add_argument('--sar_folder', help='Path to folder with SAR images (for batch mode)')
    parser.add_argument('--output', help='Output path/folder')
    parser.add_argument('--text_prompt', 
                       default='Colorise image, Region: unknown, Season: unknown',
                       help='Text prompt for conditioning')
    parser.add_argument('--region', default='unknown', help='Region description')
    parser.add_argument('--season', default='unknown', help='Season description')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                       help='Classifier-free guidance scale')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    if args.mode == 'single':
        if not args.sar_image:
            print("Error: --sar_image required for single mode")
            return
        
        text_prompt = f"Colorise image, Region: {args.region}, Season: {args.season}"
        output_path = args.output or f"{os.path.splitext(args.sar_image)[0]}_colorized.png"
        
        colorize_sar_image(args.sar_image, text_prompt, config, output_path, args.guidance_scale)
        
    elif args.mode == 'batch':
        if not args.sar_folder:
            print("Error: --sar_folder required for batch mode")
            return
        
        output_folder = args.output or "batch_inference_results"
        batch_inference(args.sar_folder, output_folder, config, 
                       args.region, args.season, args.guidance_scale)
        
    elif args.mode == 'interactive':
        interactive_inference(config)


if __name__ == '__main__':
    main()
