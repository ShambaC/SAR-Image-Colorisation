#!/usr/bin/env python3
"""
VAE Training Script for SAR and Optical Images
Trains a Variational Autoencoder to learn good latent representations
for both SAR and optical satellite images before diffusion training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
import json
from datetime import datetime
from tqdm import tqdm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils
import matplotlib.pyplot as plt

from dataset import SARDataset, create_dataloaders
from encoder import VAE_Encoder
from decoder import VAE_Decoder


class ProperVAE(nn.Module):
    """VAE using the existing VAE_Encoder and VAE_Decoder classes with proper handling"""
    
    def __init__(self, latent_dim=4, beta=1.0):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Use the existing VAE components
        self.encoder = VAE_Encoder()
        self.decoder = VAE_Decoder()
    
    def encode(self, x):
        """
        Encode images to latent space - properly handles the encoder's reparameterization
        Returns mean, logvar for VAE loss computation, and final latents for decoding
        """
        batch_size, _, height, width = x.shape
        latent_height, latent_width = height // 8, width // 8
        
        # Generate noise for the encoder's reparameterization
        noise = torch.randn(batch_size, 4, latent_height, latent_width, device=x.device)
        
        # We need to extract mean and logvar before reparameterization
        # Let's modify the encoder forward pass to get intermediate results
        x_encoded = x
        for module in self.encoder:
            if getattr(module, 'stride', None) == (2, 2):
                x_encoded = F.pad(x_encoded, (0, 1, 0, 1))
            x_encoded = module(x_encoded)
        
        # Extract mean and log_variance from the 8-channel output
        mean, log_variance = torch.chunk(x_encoded, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        
        # Apply reparameterization trick
        variance = log_variance.exp()
        stdev = variance.sqrt()
        latents = mean + stdev * noise
        
        # Apply scaling
        latents = latents * 0.18215
        
        return mean, log_variance, latents
    
    def decode(self, z):
        """Decode latent vectors to images"""
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass"""
        mean, logvar, latents = self.encode(x)
        reconstruction = self.decode(latents)
        return reconstruction, mean, logvar
    
    def compute_loss(self, x, reconstruction, mean, logvar):
        """Compute VAE loss (ELBO)"""
        batch_size = x.shape[0]
        
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, x, reduction='sum') / batch_size
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / batch_size
        
        # Total VAE loss (ELBO)
        total_loss = recon_loss + self.beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }


class VAETrainer:
    """Trainer for VAE Model"""
    
    def __init__(
        self,
        config: dict,
        dataset_path: str,
        save_dir: str = "./checkpoints/vae",
        log_dir: str = "./logs/vae"
    ):
        self.config = config
        self.dataset_path = dataset_path
        self.save_dir = save_dir
        self.log_dir = log_dir
        
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = self._setup_model()
        
        # Setup data loaders
        self.train_loader, self.val_loader = self._setup_data_loaders()
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup tensorboard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(os.path.join(log_dir, f"vae_training_{timestamp}"))
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
    
    def _setup_model(self) -> ProperVAE:
        """Setup VAE model"""
        vae_config = self.config.get("vae", {})
        
        model = ProperVAE(
            latent_dim=vae_config.get("latent_dim", 4),
            beta=vae_config.get("beta", 1.0)
        )
        
        model = model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"VAE parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        return model
    
    def _setup_data_loaders(self):
        """Setup data loaders"""
        train_loader, val_loader = create_dataloaders(
            dataset_path=self.dataset_path,
            batch_size=self.config.get("training", {}).get("batch_size", 8),
            train_split=self.config.get("data", {}).get("train_split", 0.8),
            num_workers=self.config.get("data", {}).get("num_workers", 4),
            seed=self.config.get("data", {}).get("seed", 42)
        )
        
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        return train_loader, val_loader
    
    def _setup_optimizer(self):
        """Setup optimizer"""
        training_config = self.config.get("training", {})
        
        return optim.AdamW(
            self.model.parameters(),
            lr=training_config.get("learning_rate", 1e-4),
            weight_decay=training_config.get("weight_decay", 0.01)
        )
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        training_config = self.config.get("training", {})
        
        return optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=training_config.get("epochs", 100)
        )
    
    def train_epoch(self) -> dict:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Training VAE Epoch {self.current_epoch}",
            unit="batch",
            leave=True,
            dynamic_ncols=True,
            ascii=True
        )
        
        for batch in progress_bar:
            # We'll train on both SAR and optical images
            sar_images = batch['s1_image'].to(self.device)
            optical_images = batch['s2_image'].to(self.device)
            
            # Normalize to [-1, 1] for better training
            sar_images = sar_images * 2.0 - 1.0
            optical_images = optical_images * 2.0 - 1.0
            
            # Train on SAR images
            self.optimizer.zero_grad()
            
            sar_recon, sar_mean, sar_logvar = self.model(sar_images)
            sar_losses = self.model.compute_loss(sar_images, sar_recon, sar_mean, sar_logvar)
            
            # Train on optical images
            optical_recon, optical_mean, optical_logvar = self.model(optical_images)
            optical_losses = self.model.compute_loss(optical_images, optical_recon, optical_mean, optical_logvar)
            
            # Combined loss
            total_batch_loss = sar_losses['total_loss'] + optical_losses['total_loss']
            total_recon_batch = sar_losses['recon_loss'] + optical_losses['recon_loss']
            total_kl_batch = sar_losses['kl_loss'] + optical_losses['kl_loss']
            
            # Normalize by batch size and number of image types
            batch_size = sar_images.shape[0]
            loss = total_batch_loss / (batch_size * 2)  # 2 for SAR + optical
            
            loss.backward()
            
            # Gradient clipping
            max_grad_norm = self.config.get("training", {}).get("max_grad_norm", 1.0)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_recon_loss += total_recon_batch.item() / (batch_size * 2)
            total_kl_loss += total_kl_batch.item() / (batch_size * 2)
            num_batches += 1
            self.global_step += 1
            
            # Log to tensorboard
            log_interval = self.config.get("training", {}).get("log_interval", 100)
            if self.global_step % log_interval == 0:
                self.writer.add_scalar("Loss/Train_Total", loss.item(), self.global_step)
                self.writer.add_scalar("Loss/Train_Reconstruction", total_recon_batch.item() / (batch_size * 2), self.global_step)
                self.writer.add_scalar("Loss/Train_KL", total_kl_batch.item() / (batch_size * 2), self.global_step)
                self.writer.add_scalar("Learning_Rate", self.optimizer.param_groups[0]['lr'], self.global_step)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon': f"{total_recon_batch.item() / (batch_size * 2):.4f}",
                'kl': f"{total_kl_batch.item() / (batch_size * 2):.4f}"
            })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_recon = total_recon_loss / num_batches if num_batches > 0 else 0.0
        avg_kl = total_kl_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            "train_loss": avg_loss,
            "train_recon_loss": avg_recon,
            "train_kl_loss": avg_kl
        }
    
    def validate(self) -> dict:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(
                self.val_loader,
                desc="Validating VAE",
                unit="batch",
                leave=True,
                dynamic_ncols=True,
                ascii=True
            )
            
            for batch in progress_bar:
                sar_images = batch['s1_image'].to(self.device)
                optical_images = batch['s2_image'].to(self.device)
                
                # Normalize to [-1, 1] for better training
                sar_images = sar_images * 2.0 - 1.0
                optical_images = optical_images * 2.0 - 1.0
                
                # Validate on SAR images
                sar_recon, sar_mean, sar_logvar = self.model(sar_images)
                sar_losses = self.model.compute_loss(sar_images, sar_recon, sar_mean, sar_logvar)
                
                # Validate on optical images
                optical_recon, optical_mean, optical_logvar = self.model(optical_images)
                optical_losses = self.model.compute_loss(optical_images, optical_recon, optical_mean, optical_logvar)
                
                # Combined loss
                total_batch_loss = sar_losses['total_loss'] + optical_losses['total_loss']
                total_recon_batch = sar_losses['recon_loss'] + optical_losses['recon_loss']
                total_kl_batch = sar_losses['kl_loss'] + optical_losses['kl_loss']
                
                batch_size = sar_images.shape[0]
                loss = total_batch_loss / (batch_size * 2)
                
                total_loss += loss.item()
                total_recon_loss += total_recon_batch.item() / (batch_size * 2)
                total_kl_loss += total_kl_batch.item() / (batch_size * 2)
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'val_loss': f"{loss.item():.4f}",
                    'recon': f"{total_recon_batch.item() / (batch_size * 2):.4f}",
                    'kl': f"{total_kl_batch.item() / (batch_size * 2):.4f}"
                })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_recon = total_recon_loss / num_batches if num_batches > 0 else 0.0
        avg_kl = total_kl_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            "val_loss": avg_loss,
            "val_recon_loss": avg_recon,
            "val_kl_loss": avg_kl
        }
    
    def save_checkpoint(self, metrics: dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'metrics': metrics
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, f"vae_epoch_{self.current_epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.save_dir, "vae_best.pt")
            torch.save(checkpoint, best_path)
            print(f"New best VAE model saved: {best_path}")
        
        print(f"VAE checkpoint saved: {checkpoint_path}")
    
    def save_reconstruction_samples(self, num_samples=4):
        """Save reconstruction samples for visual inspection"""
        self.model.eval()
        
        with torch.no_grad():
            # Get a batch from validation set
            batch = next(iter(self.val_loader))
            sar_images = batch['s1_image'][:num_samples].to(self.device)
            optical_images = batch['s2_image'][:num_samples].to(self.device)
            
            # Normalize inputs
            sar_images = sar_images * 2.0 - 1.0
            optical_images = optical_images * 2.0 - 1.0
            
            # Generate reconstructions
            sar_recon, _, _ = self.model(sar_images)
            optical_recon, _, _ = self.model(optical_images)
            
            # Save images
            save_dir = os.path.join(self.save_dir, "samples")
            os.makedirs(save_dir, exist_ok=True)
            
            # Convert to numpy and denormalize for display
            sar_orig = ((sar_images + 1) * 0.5).clamp(0, 1).cpu().numpy()
            sar_rec = ((sar_recon + 1) * 0.5).clamp(0, 1).cpu().numpy()
            opt_orig = ((optical_images + 1) * 0.5).clamp(0, 1).cpu().numpy()
            opt_rec = ((optical_recon + 1) * 0.5).clamp(0, 1).cpu().numpy()
            
            # Create comparison plots
            fig, axes = plt.subplots(4, 4, figsize=(16, 16))
            fig.suptitle(f'VAE Reconstructions - Epoch {self.current_epoch}')
            
            for i in range(num_samples):
                # SAR original
                axes[i, 0].imshow(np.transpose(sar_orig[i], (1, 2, 0)))
                axes[i, 0].set_title(f'SAR Original {i+1}')
                axes[i, 0].axis('off')
                
                # SAR reconstruction
                axes[i, 1].imshow(np.transpose(sar_rec[i], (1, 2, 0)))
                axes[i, 1].set_title(f'SAR Recon {i+1}')
                axes[i, 1].axis('off')
                
                # Optical original
                axes[i, 2].imshow(np.transpose(opt_orig[i], (1, 2, 0)))
                axes[i, 2].set_title(f'Optical Original {i+1}')
                axes[i, 2].axis('off')
                
                # Optical reconstruction
                axes[i, 3].imshow(np.transpose(opt_rec[i], (1, 2, 0)))
                axes[i, 3].set_title(f'Optical Recon {i+1}')
                axes[i, 3].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'reconstructions_epoch_{self.current_epoch}.png'))
            plt.close()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        print(f"VAE checkpoint loaded: {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch}")
    
    def train(self, resume_from: str = None):
        """Main training loop"""
        if resume_from:
            self.load_checkpoint(resume_from)
        
        best_val_loss = float('inf')
        total_epochs = self.config.get("training", {}).get("epochs", 50)
        
        # Create epoch progress bar
        epoch_progress = tqdm(
            range(self.current_epoch, total_epochs),
            desc="VAE Training Progress",
            unit="epoch",
            position=0,
            leave=True,
            dynamic_ncols=True,
            ascii=True
        )
        
        for epoch in epoch_progress:
            self.current_epoch = epoch
            
            print(f"\nVAE Epoch {epoch}/{total_epochs}")
            print("-" * 50)
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            
            # Log to tensorboard
            for key, value in metrics.items():
                self.writer.add_scalar(f"Epoch/{key}", value, epoch)
            
            # Print metrics
            print(f"Train Loss: {metrics['train_loss']:.4f}")
            print(f"Train Recon: {metrics['train_recon_loss']:.4f}")
            print(f"Train KL: {metrics['train_kl_loss']:.4f}")
            print(f"Val Loss: {metrics['val_loss']:.4f}")
            print(f"Val Recon: {metrics['val_recon_loss']:.4f}")
            print(f"Val KL: {metrics['val_kl_loss']:.4f}")
            
            # Update epoch progress bar
            epoch_progress.set_postfix({
                'train_loss': f"{metrics['train_loss']:.4f}",
                'val_loss': f"{metrics['val_loss']:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Save checkpoint
            is_best = metrics['val_loss'] < best_val_loss
            if is_best:
                best_val_loss = metrics['val_loss']
            
            save_every = self.config.get("training", {}).get("save_every", 10)
            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(metrics, is_best)
            
            # Save reconstruction samples
            if epoch % save_every == 0:
                self.save_reconstruction_samples()
        
        print("\nVAE training completed!")
        self.writer.close()


def get_default_vae_config():
    """Get default VAE training configuration"""
    return {
        "vae": {
            "latent_dim": 4,
            "beta": 1.0  # Beta-VAE weight for KL divergence
        },
        "training": {
            "batch_size": 64,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "epochs": 2,
            "save_every": 10,
            "max_grad_norm": 1.0,
            "log_interval": 50
        },
        "data": {
            "train_split": 0.8,
            "num_workers": 4,
            "pin_memory": True,
            "seed": 42
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Train VAE for SAR and Optical Images")
    parser.add_argument("--dataset_path", type=str, default="../Dataset",
                       help="Path to the dataset folder")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config JSON file")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/vae",
                       help="Directory to save VAE checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs/vae",
                       help="Directory for tensorboard logs")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_vae_config()
    
    # Save configuration
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "vae_config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    print("VAE Training Configuration:")
    for section, params in config.items():
        print(f"{section}:")
        for key, value in params.items():
            print(f"  {key}: {value}")
    
    # Initialize trainer
    trainer = VAETrainer(
        config=config,
        dataset_path=args.dataset_path,
        save_dir=args.save_dir,
        log_dir=args.log_dir
    )
    
    # Start training
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
