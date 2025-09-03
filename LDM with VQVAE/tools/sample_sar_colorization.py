import yaml
import argparse
import torch
import os
from torchvision import transforms
from PIL import Image
from models.unet_cond_base import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils.text_utils import get_tokenizer_and_model, get_text_representation
from utils.config_utils import get_config_value, validate_text_config, validate_image_config
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample(args):
    # Read the config file
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    
    # Extract configurations
    diffusion_config = config.get('diffusion_params', {})
    dataset_config = config.get('dataset_params', {})
    diffusion_model_config = config.get('ldm_params', {})
    autoencoder_model_config = config.get('autoencoder_params', {})
    train_config = config.get('train_params', {})

    # Create noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config.get('num_timesteps', 1000),
                                     beta_start=diffusion_config.get('beta_start', 0.0001),
                                     beta_end=diffusion_config.get('beta_end', 0.02))

    # Load VQ-VAE model
    vae = VQVAE(im_channels=dataset_config.get('im_channels', 3),
                model_config=autoencoder_model_config).to(device)
    vae.load_state_dict(torch.load(os.path.join(train_config.get('task_name', 'sar_colorization'),
                                                train_config.get('vqvae_autoencoder_ckpt_name', 'vqvae.pth')),
                                   map_location=device))
    vae.eval()

    # Load LDM model (U-Net)
    model = Unet(im_channels=autoencoder_model_config.get('z_channels', 256),
                 model_config=diffusion_model_config).to(device)
    model.load_state_dict(torch.load(os.path.join(train_config.get('task_name', 'sar_colorization'),
                                                  train_config.get('ldm_ckpt_name', 'ldm.pth')),
                                     map_location=device))
    model.eval()

    # Prepare text conditioning
    text_tokenizer, text_model = None, None
    empty_text_embed = None
    condition_config = diffusion_model_config.get('condition_config')
    if 'text' in condition_config.get('condition_types', []):
        validate_text_config(condition_config)
        text_tokenizer, text_model = get_tokenizer_and_model(
            condition_config['text_condition_config']['text_embed_model'], device=device
        )
        empty_text_embed = get_text_representation([''], text_tokenizer, text_model, device)

    # Prepare image conditioning
    gray_transform = transforms.Compose([
        transforms.Resize((dataset_config.get('im_size', 256), dataset_config.get('im_size', 256))),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    gray_image = Image.open(args.gray_image_path).convert('L')
    gray_image_tensor = gray_transform(gray_image).unsqueeze(0).to(device)

    # Generate text prompt
    text_prompt = f"Colorize the image, Region: {args.region}, Season: {args.season}"
    
    with torch.no_grad():
        # Get text representation
        text_condition = get_text_representation([text_prompt], text_tokenizer, text_model, device)

        cond_input = {'text': text_condition, 'image': gray_image_tensor}

        # Sampling loop
        x = torch.randn(1, autoencoder_model_config.get('z_channels', 256), 
                        dataset_config.get('im_size', 256) // (2**len(autoencoder_model_config.get('down_sampling_levels', []))), 
                        dataset_config.get('im_size', 256) // (2**len(autoencoder_model_config.get('down_sampling_levels', [])))).to(device)

        for t in tqdm(reversed(range(scheduler.num_timesteps))):
            t_tensor = torch.full((1,), t, dtype=torch.long, device=device)
            
            # Classifier-Free Guidance
            # Predict noise with conditioning
            noise_pred_cond = model(x, t_tensor, cond_input)
            
            # Predict noise without text conditioning (but with image conditioning)
            uncond_input = {'text': empty_text_embed, 'image': gray_image_tensor}
            noise_pred_uncond = model(x, t_tensor, uncond_input)
            
            # Combine predictions
            guidance_scale = train_config.get('guidance_scale', 7.5)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # Denoise
            x = scheduler.denoise(x, noise_pred, t)

        # Decode the generated latent
        decoded_image = vae.decode(x)

    # Post-process and save the image
    decoded_image = torch.clamp(decoded_image, -1., 1.).squeeze(0)
    decoded_image = (decoded_image + 1) / 2
    
    output_transform = transforms.ToPILImage()
    pil_image = output_transform(decoded_image.cpu())
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    pil_image.save(args.output_path)
    print(f"Saved generated image to {args.output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for SAR Colorization Inference')
    parser.add_argument('--config', dest='config_path', default='config/sar_colorization.yaml', type=str)
    parser.add_argument('--gray_image_path', dest='gray_image_path', required=True, type=str, help="Path to the input grayscale SAR image.")
    parser.add_argument('--region', dest='region', required=True, type=str, choices=['tropical', 'temperate', 'arctic'], help="Region of the image.")
    parser.add_argument('--season', dest='season', required=True, type=str, choices=['winter', 'summer', 'spring', 'fall'], help="Season of the image.")
    parser.add_argument('--output_path', dest='output_path', default='output/generated_image.png', type=str, help="Path to save the generated color image.")
    args = parser.parse_args()
    sample(args)
