import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import csv

from transformers import CLIPTokenizer
from diffusion import Diffusion
from clip import CLIP
from encoder import VAE_Encoder
from dataset import SARTrainDataset
from ddpm import DDPMSampler

def train(config):
    # Setup
    device = config['device']
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # --- Logging Setup ---
    log_file = open(config['diffusion_log_path'], 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['epoch', 'avg_loss'])
    # ---------------------
    
    # Load Models
    # 1. VAE (Encoder) - Trained in train_vae.py
    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(torch.load(os.path.join(output_dir, "vae_encoder.pt"), map_location=device))
    encoder.eval() # Set to evaluation mode and freeze weights
    for param in encoder.parameters():
        param.requires_grad = False
        
    # 2. CLIP Text Encoder and Tokenizer
    tokenizer = CLIPTokenizer("./data/vocab.json", merges_file="./data/merges.txt")
    clip_model = CLIP().to(device)
    # The model loader script handles loading from the large .ckpt
    # We will assume CLIP weights are loaded into the diffusion model's state_dict loading for simplicity
    # or loaded separately if needed. For now, we assume it's ready.
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False

    # 3. Diffusion Model (U-Net)
    # MODIFICATION: The U-Net's first conv layer must accept 8 channels:
    # 4 for the noisy target latent + 4 for the conditional SAR latent.
    diffusion_model = Diffusion().to(device)
    diffusion_model.unet.encoders[0][0] = torch.nn.Conv2d(8, 320, kernel_size=3, padding=1).to(device)
    diffusion_model.train() # Set to training mode

    # Dataset
    dataset = SARTrainDataset(
        data_root=config['data_root'],
        image_size=(config['image_height'], config['image_width'])
    )
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    
    # Sampler
    sampler = DDPMSampler(torch.Generator(device=device))

    # Optimizer
    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=config['diffusion_train']['lr'])
    
    # Loss Function
    loss_fn = torch.nn.MSELoss()
    
    print("Starting Diffusion model training...")
    for epoch in range(config['diffusion_train']['epochs']):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        total_loss = 0.0

        # --- Loss accumulator for logging ---
        epoch_total_loss = 0.0
        # ------------------------------------

        for batch in progress_bar:
            optimizer.zero_grad()

            s1_images = batch['s1_image'].to(device) # Grayscale condition
            s2_images = batch['s2_image'].to(device) # Color target
            prompts = batch['prompt']

            # 1. Encode target image into latent space with VAE
            # Use no_grad as VAE is frozen
            with torch.no_grad():
                # Shape: (Batch_Size, 4, Height/8, Width/8)
                latent_noise_shape = (
                    s2_images.shape[0], 
                    4, 
                    config['image_height'] // 8, 
                    config['image_width'] // 8
                )
                encoder_noise = torch.randn(latent_noise_shape, device=device)

                target_latents = encoder(s2_images, encoder_noise)

                # Also encode the conditional (grayscale) image
                condition_latents = encoder(s1_images, encoder_noise)

            # 2. Sample a random timestep and add noise to target latents
            timesteps = torch.randint(0, sampler.num_train_timesteps, (target_latents.shape[0],), device=device)
            noise = torch.randn_like(target_latents)
            noisy_latents = sampler.add_noise(target_latents, timesteps)

            # 3. Get text embeddings from CLIP
            with torch.no_grad():
                tokens = tokenizer.batch_encode_plus(
                    prompts, padding="max_length", max_length=77, return_tensors="pt"
                ).input_ids.to(device)
                text_embeddings = clip_model(tokens)
            
            # 4. Concatenate noisy latent with condition latent
            model_input = torch.cat([noisy_latents, condition_latents], dim=1)
            
            # 5. Get time embeddings
            time_embeddings = torch.cat([get_time_embedding(t).to(device) for t in timesteps])
            
            # 6. Predict noise using the U-Net
            predicted_noise = diffusion_model(model_input, text_embeddings, time_embeddings)
            
            # 7. Calculate loss
            loss = loss_fn(predicted_noise, noise)
            
            # 8. Backpropagate and update weights
            loss.backward()
            optimizer.step()

            # --- Accumulate loss ---
            epoch_total_loss += loss.item()
            # ----------------------

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        # --- Calculate and log average epoch loss ---
        avg_loss = epoch_total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        log_writer.writerow([epoch + 1, avg_loss])
        log_file.flush()
        # -----------------------------------------

        # Save a checkpoint periodically
        if (epoch + 1) % 10 == 0:
            torch.save(diffusion_model.state_dict(), os.path.join(output_dir, f"diffusion_epoch_{epoch+1}.pt"))

    # --- Close the log file ---
    log_file.close()
    # -------------------------

    torch.save(diffusion_model.state_dict(), config['diffusion_ckpt_path'])
    print(f"Diffusion model saved to {config['diffusion_ckpt_path']}")


# You need the get_time_embedding function from pipeline.py
def get_time_embedding(timestep):
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)