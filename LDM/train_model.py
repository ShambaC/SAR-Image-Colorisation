import yaml
import argparse
import numpy as np
import torch
import os
from tqdm import tqdm
from torch.optim import Adam
from dataset.sar_dataset import SARDataset
from torch.utils.data import DataLoader
from models.unet_cond_base import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils.text_utils import *
from utils.config_utils import *
from utils.diffusion_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_autoencoder(config):
    """
    Train the VQVAE autoencoder on optical (Sentinel-2) images
    """
    print("=== Training Autoencoder (VQVAE) ===")
    
    dataset_config = config['dataset_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
      # Create dataset for autoencoder training (only optical images needed)
    dataset = SARDataset(split='train',
                        im_path=dataset_config['im_path'],
                        im_size=dataset_config['im_size'],
                        im_channels=dataset_config['im_channels'],
                        use_latents=False,
                        condition_config=None,  # No conditioning for autoencoder
                        train_split=dataset_config.get('train_split', 0.7),
                        val_split=dataset_config.get('val_split', 0.15),
                        test_split=dataset_config.get('test_split', 0.15),
                        random_seed=train_config.get('seed', 42))
    
    data_loader = DataLoader(dataset,
                           batch_size=train_config['autoencoder_batch_size'],
                           shuffle=True,
                           num_workers=2)
    
    # Create models
    vqvae = VQVAE(im_channels=dataset_config['im_channels'],
                  model_config=autoencoder_model_config).to(device)
    
    # Optimizer
    optimizer = Adam(vqvae.parameters(), lr=train_config['autoencoder_lr'])
    criterion = torch.nn.MSELoss()
    
    # Create task directory
    os.makedirs(train_config['task_name'], exist_ok=True)
    
    # Training loop
    num_epochs = train_config['autoencoder_epochs']
    
    for epoch_idx in range(num_epochs):
        vqvae.train()
        losses = []
        
        for batch_idx, data in enumerate(tqdm(data_loader, desc=f"Epoch {epoch_idx+1}/{num_epochs}")):
            # For autoencoder training, we only need the optical images (targets)
            if isinstance(data, tuple):
                images = data[0]  # Take only the image, ignore conditioning
            else:
                images = data
            
            images = images.float().to(device)
            
            optimizer.zero_grad()
              # Forward pass through autoencoder
            model_output = vqvae(images)
            output, z, vq_losses = model_output
            
            # Calculate reconstruction loss
            recon_loss = criterion(output, images)
            
            # Total loss including VQ losses
            total_loss = (recon_loss + 
                         train_config['codebook_weight'] * vq_losses['codebook_loss'] +
                         train_config['commitment_beta'] * vq_losses['commitment_loss'])
            
            losses.append(total_loss.item())
            total_loss.backward()
            optimizer.step()
            
            # Save sample images periodically
            if batch_idx % train_config['autoencoder_img_save_steps'] == 0:
                save_sample_images(images, output, epoch_idx, batch_idx, train_config)
        
        avg_loss = np.mean(losses)
        print(f'Epoch {epoch_idx + 1}/{num_epochs} | Loss: {avg_loss:.4f}')
        
        # Save checkpoint
        torch.save(vqvae.state_dict(), 
                  os.path.join(train_config['task_name'], train_config['vqvae_autoencoder_ckpt_name']))
    
    print("Autoencoder training completed!")
    
    # Optionally save latents for faster LDM training
    if train_config['save_latents']:
        save_latents_for_dataset(vqvae, dataset, train_config)


def save_sample_images(original, reconstructed, epoch, batch_idx, train_config):
    """Save sample reconstruction images"""
    import torchvision.utils as vutils
    
    # Denormalize images from [-1, 1] to [0, 1]
    original = (original + 1) / 2
    reconstructed = (reconstructed + 1) / 2
    
    # Create comparison grid
    comparison = torch.cat([original[:4], reconstructed[:4]], dim=0)
    
    samples_dir = os.path.join(train_config['task_name'], 'autoencoder_samples')
    os.makedirs(samples_dir, exist_ok=True)
    
    vutils.save_image(comparison, 
                     os.path.join(samples_dir, f'epoch_{epoch}_batch_{batch_idx}.png'),
                     nrow=4, normalize=True)


def save_latents_for_dataset(vqvae, dataset, train_config):
    """Save encoded latents for faster LDM training"""
    print("Saving latents for faster LDM training...")
    
    vqvae.eval()
    latents_dir = os.path.join(train_config['task_name'], train_config['vqvae_latent_dir_name'])
    os.makedirs(latents_dir, exist_ok=True)
    
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for idx, data in enumerate(tqdm(data_loader, desc="Encoding latents")):
            if isinstance(data, tuple):
                image = data[0]
            else:
                image = data
                
            image = image.float().to(device)
            latent, _ = vqvae.encode(image)
            
            # Save latent with image path as key
            image_path = dataset.s2_images[idx]
            latent_path = os.path.join(latents_dir, f'latent_{idx}.pt')
            torch.save({'path': image_path, 'latent': latent.cpu()}, latent_path)


def train_ldm(config):
    """
    Train the Latent Diffusion Model with text and image conditioning
    """
    print("=== Training Latent Diffusion Model ===")
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
    
    # Create noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                   beta_start=diffusion_config['beta_start'],
                                   beta_end=diffusion_config['beta_end'])
    
    # Setup text conditioning
    text_tokenizer = None
    text_model = None
    empty_text_embed = None
    condition_config = diffusion_model_config['condition_config']
    condition_types = condition_config['condition_types']
    
    if 'text' in condition_types:
        validate_text_config(condition_config)
        with torch.no_grad():
            text_tokenizer, text_model = get_tokenizer_and_model(
                condition_config['text_condition_config']['text_embed_model'], 
                device=device)
            empty_text_embed = get_text_representation([''], text_tokenizer, text_model, device)
      # Create dataset
    dataset = SARDataset(split='train',
                        im_path=dataset_config['im_path'],
                        im_size=dataset_config['im_size'],
                        im_channels=dataset_config['im_channels'],
                        use_latents=train_config['save_latents'],
                        latent_path=os.path.join(train_config['task_name'],
                                               train_config['vqvae_latent_dir_name']),
                        condition_config=condition_config,
                        train_split=dataset_config.get('train_split', 0.7),
                        val_split=dataset_config.get('val_split', 0.15),
                        test_split=dataset_config.get('test_split', 0.15),
                        random_seed=train_config.get('seed', 42))
    
    data_loader = DataLoader(dataset,
                           batch_size=train_config['ldm_batch_size'],
                           shuffle=True,
                           num_workers=2)
    
    # Create diffusion model
    model = Unet(im_channels=autoencoder_model_config['z_channels'],
                model_config=diffusion_model_config).to(device)
    model.train()
    
    # Load VQVAE if not using latents
    vqvae = None
    if not dataset.use_latents:
        print('Loading VQVAE model as latents not present')
        vqvae = VQVAE(im_channels=dataset_config['im_channels'],
                     model_config=autoencoder_model_config).to(device)
        vqvae.eval()
        
        vqvae_path = os.path.join(train_config['task_name'], train_config['vqvae_autoencoder_ckpt_name'])
        if os.path.exists(vqvae_path):
            print('Loaded VQVAE checkpoint')
            vqvae.load_state_dict(torch.load(vqvae_path, map_location=device))
        else:
            raise Exception('VQVAE checkpoint not found. Please train autoencoder first.')
        
        # Freeze VQVAE parameters
        for param in vqvae.parameters():
            param.requires_grad = False
    
    # Optimizer and loss
    optimizer = Adam(model.parameters(), lr=train_config['ldm_lr'])
    criterion = torch.nn.MSELoss()
    
    # Training loop
    num_epochs = train_config['ldm_epochs']
    
    for epoch_idx in range(num_epochs):
        losses = []
        
        for data in tqdm(data_loader, desc=f"LDM Epoch {epoch_idx+1}/{num_epochs}"):
            target_image, cond_input = data
            
            optimizer.zero_grad()
            target_image = target_image.float().to(device)
            
            # Encode target image to latent space if needed
            if not dataset.use_latents:
                with torch.no_grad():
                    target_image, _ = vqvae.encode(target_image)
            
            # Process conditioning inputs
            if 'text' in condition_types:
                with torch.no_grad():
                    text_condition = get_text_representation(cond_input['text'],
                                                           text_tokenizer,
                                                           text_model,
                                                           device)
                    text_drop_prob = condition_config['text_condition_config'].get('cond_drop_prob', 0.)
                    text_condition = drop_text_condition(text_condition, target_image, empty_text_embed, text_drop_prob)
                    cond_input['text'] = text_condition
            
            if 'image' in condition_types:
                sar_image = cond_input['image'].to(device)
                image_drop_prob = condition_config['image_condition_config'].get('cond_drop_prob', 0.)
                cond_input['image'] = drop_image_condition(sar_image, target_image, image_drop_prob)
            
            # Sample noise and timestep
            noise = torch.randn_like(target_image).to(device)
            t = torch.randint(0, diffusion_config['num_timesteps'], (target_image.shape[0],)).to(device)
            
            # Add noise to target image
            noisy_image = scheduler.add_noise(target_image, noise, t)
            
            # Predict noise
            noise_pred = model(noisy_image, t, cond_input=cond_input)
            loss = criterion(noise_pred, noise)
            
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        
        avg_loss = np.mean(losses)
        print(f'LDM Epoch {epoch_idx + 1}/{num_epochs} | Loss: {avg_loss:.4f}')
        
        # Save checkpoint
        torch.save(model.state_dict(), 
                  os.path.join(train_config['task_name'], train_config['ldm_ckpt_name']))
    
    print("LDM training completed!")


def main():
    parser = argparse.ArgumentParser(description='Train SAR to Optical Image Translation Model')
    parser.add_argument('--config', dest='config_path',
                       default='config/sar_config.yaml', type=str,
                       help='Path to configuration file')
    parser.add_argument('--stage', choices=['autoencoder', 'ldm', 'both'],
                       default='both', help='Training stage')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            return
    
    print("Configuration loaded:")
    print(yaml.dump(config, default_flow_style=False))
    
    # Set random seeds for reproducibility
    torch.manual_seed(config['train_params']['seed'])
    np.random.seed(config['train_params']['seed'])
    
    # Training stages
    if args.stage in ['autoencoder', 'both']:
        train_autoencoder(config)
    
    if args.stage in ['ldm', 'both']:
        train_ldm(config)
    
    print("Training completed successfully!")


if __name__ == '__main__':
    main()
