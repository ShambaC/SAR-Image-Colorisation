import os
import torch
import numpy as np
from PIL import Image
import argparse
import yaml
from tqdm import tqdm
from transformers import CLIPTokenizer

from pipeline import get_time_embedding, rescale
from ddpm import DDPMSampler
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion
from clip import CLIP
import torchvision.transforms as T

def inference(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = config['device']

    # Load Models
    print("Loading models...")
    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(torch.load(os.path.join(config['output_dir'], "vae_encoder.pt"), map_location=device))

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(torch.load(os.path.join(config['output_dir'], "vae_decoder.pt"), map_location=device))

    diffusion_model = Diffusion().to(device)
    # Remember to modify the input layer to match the trained model
    diffusion_model.unet.encoders[0][0] = torch.nn.Conv2d(8, 320, kernel_size=3, padding=1).to(device)
    diffusion_model.load_state_dict(torch.load(config['diffusion_ckpt_path'], map_location=device))
    
    clip_model = CLIP().to(device)
    # In a real scenario, you'd load pretrained CLIP weights here.
    # We assume they are available for the process.
    
    tokenizer = CLIPTokenizer("./data/vocab.json", merges_file="./data/merges.txt")
    
    # Prepare inputs
    prompt = f"Colorize the image, Region: {args.region}, Season: {args.season}"
    uncond_prompt = "" # For classifier-free guidance
    
    # Load and transform the input grayscale image
    input_image = Image.open(args.input_image_path).convert("RGB")
    transform = T.Compose([
        T.Resize((config['image_height'], config['image_width'])),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    s1_tensor = transform(input_image).unsqueeze(0).to(device)
    
    sampler = DDPMSampler(torch.Generator(device=device))
    sampler.set_inference_timesteps(config['diffusion_train']['num_inference_steps'])

    latents_shape = (1, 4, config['image_height'] // 8, config['image_width'] // 8)

    with torch.no_grad():
        # --- Text Conditioning ---
        cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77, return_tensors="pt").input_ids.to(device)
        cond_context = clip_model(cond_tokens)

        uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding="max_length", max_length=77, return_tensors="pt").input_ids.to(device)
        uncond_context = clip_model(uncond_tokens)
        
        context = torch.cat([cond_context, uncond_context])

        # --- Image Conditioning ---

        latent_noise_shape = (
            s1_tensor.shape[0], 
            4, 
            config['image_height'] // 8, 
            config['image_width'] // 8
        )
        conditional_noise = torch.randn(latent_noise_shape, device=device)

        condition_latents = encoder(s1_tensor, conditional_noise)
        condition_latents_cfg = torch.cat([condition_latents, condition_latents])

        # --- Denoising Loop ---
        latents = torch.randn(latents_shape, generator=sampler.generator, device=device)
        
        for i, timestep in enumerate(tqdm(sampler.timesteps)):
            time_embedding = get_time_embedding(timestep).to(device)

            model_input_latents = latents.repeat(2, 1, 1, 1) # For CFG
            
            # Concatenate noisy latents with condition latents
            model_input = torch.cat([model_input_latents, condition_latents_cfg], dim=1)

            model_output = diffusion_model(model_input, context, time_embedding)
            
            output_cond, output_uncond = model_output.chunk(2)
            model_output = config['diffusion_train']['cfg_scale'] * (output_cond - output_uncond) + output_uncond
            
            latents = sampler.step(timestep, latents, model_output)

        # --- Decode final latent to image ---
        images = decoder(latents)
        
    # Rescale and save the output image
    images = rescale(images, (-1, 1), (0, 255), clamp=True)
    images = images.permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
    output_image = Image.fromarray(images[0])
    output_image.save(args.output_image_path)
    print(f"Image saved to {args.output_image_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate an image from a grayscale SAR image and text prompt.")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to the config file.")
    parser.add_argument('--input_image_path', type=str, required=True, help="Path to the input grayscale SAR image.")
    parser.add_argument('--output_image_path', type=str, default='output.png', help="Path to save the generated color image.")
    parser.add_argument('--region', type=str, required=True, choices=['tropical', 'temperate', 'arctic'], help="Region for the text prompt.")
    parser.add_argument('--season', type=str, required=True, choices=['winter', 'summer', 'spring', 'fall'], help="Season for the text prompt.")
    args = parser.parse_args()
    inference(args)