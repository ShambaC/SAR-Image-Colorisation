import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import csv

from encoder import VAE_Encoder
from decoder import VAE_Decoder
from dataset import VAEPretrainDataset

def train(config):
    # Setup
    device = config['device']
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # --- Logging Setup ---
    log_file = open(config['vae_log_path'], 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['epoch', 'avg_loss', 'avg_recon_loss', 'avg_kl_div'])
    # ---------------------

    # Models
    encoder = VAE_Encoder().to(device)
    decoder = VAE_Decoder().to(device)

    # Dataset
    dataset = VAEPretrainDataset(
        data_root=config['data_root'],
        image_size=(config['image_height'], config['image_width'])
    )
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])

    # Optimizer
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=config['vae_train']['lr'])

    print("Starting VAE training...")
    for epoch in range(config['vae_train']['epochs']):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        total_loss = 0.0

        # --- Loss accumulators for logging ---
        epoch_total_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_div = 0.0
        # ------------------------------------

        for batch in progress_bar:
            images = batch.to(device)
            
            # Forward pass
            encoder_noise = torch.randn(
                (images.shape[0], 4, config['image_height'] // 8, config['image_width'] // 8),
                device=device
            )
            # The encoder returns mean and log_variance, we need to extract them for the loss
            encoded_output = encoder.forward_for_loss(images)
            mean, log_variance = torch.chunk(encoded_output, 2, dim=1)
            
            # Reparameterization trick
            stdev = torch.exp(0.5 * log_variance)
            z = mean + stdev * encoder_noise
            
            # Scale as per the original implementation before decoding
            scaled_z = z * 0.18215
            
            reconstructed_images = decoder(scaled_z)

            # Calculate Loss
            reconstruction_loss = F.mse_loss(reconstructed_images, images)
            kl_div = -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp())
            loss = reconstruction_loss + config['vae_train']['kl_weight'] * kl_div

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --- Accumulate losses ---
            epoch_total_loss += loss.item()
            epoch_recon_loss += reconstruction_loss.item()
            epoch_kl_div += kl_div.item()
            # ------------------------
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item(), recon_loss=reconstruction_loss.item(), kl_div=kl_div.item())

        # --- Calculate and log average epoch losses ---
        avg_total_loss = epoch_total_loss / len(dataloader)
        avg_recon_loss = epoch_recon_loss / len(dataloader)
        avg_kl_div = epoch_kl_div / len(dataloader)

        print(f"Epoch {epoch+1} Average Loss: {avg_total_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}, KL Div: {avg_kl_div:.4f}")
        log_writer.writerow([epoch + 1, avg_total_loss, avg_recon_loss, avg_kl_div])
        log_file.flush() # Ensure data is written to the file
        # -------------------------------------------

    # --- Close the log file ---
    log_file.close()
    # -------------------------

    # Save models
    torch.save(encoder.state_dict(), os.path.join(output_dir, "vae_encoder.pt"))
    torch.save(decoder.state_dict(), os.path.join(output_dir, "vae_decoder.pt"))
    print(f"VAE models saved to {output_dir}")