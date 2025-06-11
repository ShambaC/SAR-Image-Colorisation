import yaml
import argparse
import numpy as np
import torch
import os
import json
import csv
import time
from datetime import datetime
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


def save_loss_data(loss_history, loss_type, task_name):
    """
    Save loss data in both JSON and CSV formats for easy plotting
    
    Args:
        loss_history: Dictionary containing loss data
        loss_type: 'autoencoder' or 'ldm'
        task_name: Task directory name
    """
    # Create losses directory
    losses_dir = os.path.join(task_name, 'losses')
    os.makedirs(losses_dir, exist_ok=True)
    
    # Save as JSON for easy loading
    json_path = os.path.join(losses_dir, f'{loss_type}_loss_history.json')
    with open(json_path, 'w') as f:
        json.dump(loss_history, f, indent=2)
    
    # Save as CSV for plotting with pandas/matplotlib
    csv_path = os.path.join(losses_dir, f'{loss_type}_loss_history.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        if loss_type == 'autoencoder':
            # Headers for autoencoder losses
            writer.writerow(['epoch', 'batch', 'recon_loss', 'codebook_loss', 'commitment_loss', 'total_loss', 'epoch_avg_loss', 'best_loss', 'timestamp'])
            
            for epoch_data in loss_history['epochs']:
                epoch_num = epoch_data['epoch']
                epoch_avg = epoch_data['avg_loss']
                best_loss = epoch_data['best_loss']
                timestamp = epoch_data['timestamp']
                
                for batch_data in epoch_data['batches']:
                    writer.writerow([
                        epoch_num,
                        batch_data['batch'],
                        batch_data['recon_loss'],
                        batch_data['codebook_loss'], 
                        batch_data['commitment_loss'],
                        batch_data['total_loss'],
                        epoch_avg,
                        best_loss,
                        timestamp
                    ])
        
        elif loss_type == 'ldm':
            # Headers for LDM losses
            writer.writerow(['epoch', 'batch', 'loss', 'epoch_avg_loss', 'best_loss', 'timestamp'])
            
            for epoch_data in loss_history['epochs']:
                epoch_num = epoch_data['epoch']
                epoch_avg = epoch_data['avg_loss']
                best_loss = epoch_data['best_loss']
                timestamp = epoch_data['timestamp']
                
                for batch_data in epoch_data['batches']:
                    writer.writerow([
                        epoch_num,
                        batch_data['batch'],
                        batch_data['loss'],
                        epoch_avg,
                        best_loss,
                        timestamp
                    ])
    
    print(f"Loss data saved to {json_path} and {csv_path}")


def load_loss_history(loss_type, task_name):
    """
    Load existing loss history if resuming training
    
    Args:
        loss_type: 'autoencoder' or 'ldm'
        task_name: Task directory name
    
    Returns:
        Dictionary containing loss history or empty structure
    """
    losses_dir = os.path.join(task_name, 'losses')
    json_path = os.path.join(losses_dir, f'{loss_type}_loss_history.json')
    
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except:
            print(f"Warning: Could not load existing loss history from {json_path}")
    
    # Return empty structure
    return {
        'training_started': datetime.now().isoformat(),
        'config': {},
        'epochs': []
    }


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
    
    # Check for existing checkpoint and resume if requested
    autoencoder_checkpoint_path = os.path.join(train_config['task_name'], train_config['vqvae_autoencoder_ckpt_name'])
    start_epoch = 0
    best_loss = float('inf')
    
    if os.path.exists(autoencoder_checkpoint_path) and train_config.get('resume_training', False):
        print(f'Loading autoencoder checkpoint from {autoencoder_checkpoint_path}')
        checkpoint = torch.load(autoencoder_checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Enhanced checkpoint format
            vqvae.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint.get('best_loss', float('inf'))
            print(f'Resuming autoencoder training from epoch {start_epoch}, best loss: {best_loss:.4f}')
        else:
            # Simple state dict format (backward compatibility)
            vqvae.load_state_dict(checkpoint)
            print('Loaded autoencoder weights, starting from epoch 0')
    
    # Load existing loss history if resuming
    loss_history = load_loss_history('autoencoder', train_config['task_name'])
    if loss_history['epochs']:
        start_epoch = loss_history['epochs'][-1]['epoch']
        best_loss = loss_history['epochs'][-1]['best_loss']
        print(f"Resuming from epoch {start_epoch}, best loss: {best_loss:.4f}")
    
    # Training loop
    num_epochs = train_config['autoencoder_epochs']    
    for epoch_idx in range(start_epoch, num_epochs):
        vqvae.train()
        losses = []
        epoch_loss_data = {
            'epoch': epoch_idx + 1,
            'batches': []
        }
        
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
            
            # Save loss data for this batch
            batch_loss_data = {
                'batch': batch_idx,
                'recon_loss': recon_loss.item(),
                'codebook_loss': vq_losses['codebook_loss'].item(),
                'commitment_loss': vq_losses['commitment_loss'].item(),
                'total_loss': total_loss.item()
            }
            epoch_loss_data['batches'].append(batch_loss_data)
            
            # Save sample images periodically
            if batch_idx % train_config['autoencoder_img_save_steps'] == 0:
                save_sample_images(images, output, epoch_idx, batch_idx, train_config)
        avg_loss = np.mean(losses)
        print(f'Epoch {epoch_idx + 1}/{num_epochs} | Loss: {avg_loss:.4f}')
        
        # Update best loss
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        # Save enhanced checkpoint
        checkpoint_data = {
            'epoch': epoch_idx + 1,
            'model_state_dict': vqvae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'best_loss': best_loss
        }
        torch.save(checkpoint_data, autoencoder_checkpoint_path)
        
        # Also save simple state dict for backward compatibility
        torch.save(vqvae.state_dict(), 
                  os.path.join(train_config['task_name'], 'vqvae_simple.pth'))
        
        # Save loss history
        timestamp = datetime.now().isoformat()
        loss_history['epochs'].append({
            'epoch': epoch_idx + 1,
            'avg_loss': avg_loss,
            'best_loss': best_loss,
            'timestamp': timestamp,
            'batches': epoch_loss_data['batches']
        })
        save_loss_data(loss_history, 'autoencoder', train_config['task_name'])
    
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


def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

    print(f"LDM Model Parameters: {count_parameters(model):,}")
    
    # Load VQVAE if not using latents
    vqvae = None
    if not dataset.use_latents:
        print('Loading VQVAE model as latents not present')
        vqvae = VQVAE(im_channels=dataset_config['im_channels'],
                     model_config=autoencoder_model_config).to(device)
        vqvae.eval()
        vqvae_path = os.path.join(train_config['task_name'], train_config['vqvae_autoencoder_ckpt_name'])
        if os.path.exists(vqvae_path):
            print('Loading VQVAE checkpoint')
            checkpoint = torch.load(vqvae_path, map_location=device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Enhanced checkpoint format
                vqvae.load_state_dict(checkpoint['model_state_dict'])
                print('Loaded VQVAE from enhanced checkpoint')
            else:
                # Simple state dict format (backward compatibility)
                vqvae.load_state_dict(checkpoint)
                print('Loaded VQVAE from simple checkpoint')
        else:
            raise Exception('VQVAE checkpoint not found. Please train autoencoder first.')
        
        # Freeze VQVAE parameters
        for param in vqvae.parameters():
            param.requires_grad = False
      # Optimizer and loss
    optimizer = Adam(model.parameters(), lr=float(train_config['ldm_lr']))
    criterion = torch.nn.MSELoss()
    
    # Check for existing LDM checkpoint and resume if requested
    ldm_checkpoint_path = os.path.join(train_config['task_name'], train_config['ldm_ckpt_name'])
    start_epoch = 0
    best_loss = float('inf')
    
    if os.path.exists(ldm_checkpoint_path) and train_config.get('resume_training', False):
        print(f'Loading LDM checkpoint from {ldm_checkpoint_path}')
        checkpoint = torch.load(ldm_checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Enhanced checkpoint format
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint.get('best_loss', float('inf'))
            print(f'Resuming LDM training from epoch {start_epoch}, best loss: {best_loss:.4f}')
        else:
            # Simple state dict format (backward compatibility)
            model.load_state_dict(checkpoint)
            print('Loaded LDM weights, starting from epoch 0')
    
    # Load existing loss history if resuming
    loss_history = load_loss_history('ldm', train_config['task_name'])
    if loss_history['epochs'] and train_config.get('resume_training', False):
        start_epoch = loss_history['epochs'][-1]['epoch']
        best_loss = loss_history['epochs'][-1]['best_loss']
        print(f"Resuming from epoch {start_epoch}, best loss: {best_loss:.4f}")
      # Training loop
    num_epochs = train_config['ldm_epochs']
    
    for epoch_idx in range(start_epoch, num_epochs):
        losses = []
        epoch_loss_data = {
            'epoch': epoch_idx + 1,
            'batches': []
        }
        
        for batch_idx, data in enumerate(tqdm(data_loader, desc=f"LDM Epoch {epoch_idx+1}/{num_epochs}")):
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
            
            # Save loss data for this batch
            batch_loss_data = {
                'batch': batch_idx,
                'loss': loss.item()
            }
            epoch_loss_data['batches'].append(batch_loss_data)
        avg_loss = np.mean(losses)
        print(f'LDM Epoch {epoch_idx + 1}/{num_epochs} | Loss: {avg_loss:.4f}')
        
        # Update best loss
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        # Save enhanced checkpoint
        checkpoint_data = {
            'epoch': epoch_idx + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'best_loss': best_loss
        }
        torch.save(checkpoint_data, ldm_checkpoint_path)
        
        # Also save simple state dict for backward compatibility
        torch.save(model.state_dict(), 
                  os.path.join(train_config['task_name'], 'ldm_simple.pth'))
          # Save loss history
        timestamp = datetime.now().isoformat()
        loss_history['epochs'].append({
            'epoch': epoch_idx + 1,
            'avg_loss': avg_loss,
            'best_loss': best_loss,
            'timestamp': timestamp,
            'batches': epoch_loss_data['batches']
        })
        save_loss_data(loss_history, 'ldm', train_config['task_name'])
    
    print("LDM training completed!")


def main():
    parser = argparse.ArgumentParser(description='Train SAR to Optical Image Translation Model')
    parser.add_argument('--config', dest='config_path',
                       default='config/sar_config.yaml', type=str,
                       help='Path to configuration file')
    parser.add_argument('--stage', choices=['autoencoder', 'ldm', 'both'],
                       default='both', help='Training stage')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from existing checkpoint')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            return
    
    # Override resume setting from command line
    if args.resume:
        config['train_params']['resume_training'] = True
    
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
