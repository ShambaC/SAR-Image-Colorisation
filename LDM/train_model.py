import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
import json
from datetime import datetime
from tqdm import tqdm
import numpy as np
import math
from PIL import Image
import torchvision.transforms as transforms

from dataset import SARDataset, create_dataloaders
from tokenizer import SimpleTokenizer
from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion


def get_timestep_embedding(timesteps, embedding_dim):
    """
    Create sinusoidal timestep embeddings.
    
    Args:
        timesteps: 1-D Tensor of N indices, one per batch element.
        embedding_dim: The dimension of the output.
        
    Returns:
        Tensor of shape (N, embedding_dim)
    """
    assert len(timesteps.shape) == 1, "Timesteps should be 1-D"
    
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.cat([emb, torch.zeros(emb.shape[0], 1, device=emb.device)], dim=1)
    
    return emb


class SARColorizationModel(nn.Module):
    """Complete SAR to Optical Image Colorization Model"""
    
    def __init__(
        self,
        clip_config: dict,
        diffusion_config: dict,
        tokenizer: SimpleTokenizer
    ):
        super().__init__()
        
        self.tokenizer = tokenizer
          # CLIP Text Encoder
        self.clip = CLIP(
            n_vocab=tokenizer.get_vocab_size(),
            n_embd=clip_config.get("n_embd", 768),
            n_token=clip_config.get("max_length", 77),
            n_head=clip_config.get("n_head", 12),
            n_layers=clip_config.get("n_layers", 12)
        )
        
        # VAE Encoder (for encoding input SAR images to latent space)
        self.vae_encoder = VAE_Encoder()
        
        # VAE Decoder (for decoding latents back to images)
        self.vae_decoder = VAE_Decoder()        # U-Net for diffusion - now handles SAR+optical concatenated input (8 channels)
        self.diffusion = Diffusion(in_channels=8)
        
        # Input projection layer to match concatenated SAR+optical input (8 channels to 4 output)
        self.input_proj = nn.Conv2d(8, 4, kernel_size=1)
    
    def encode_text(self, prompts):
        """Encode text prompts using CLIP"""
        token_ids = self.tokenizer.encode_batch(prompts)
        if isinstance(token_ids, list):
            token_ids = torch.tensor(token_ids)
        token_ids = token_ids.to(next(self.parameters()).device)
        
        # Get text embeddings
        text_embeddings = self.clip(token_ids)
        
        return text_embeddings
    
    def encode_images(self, images, noise=None):
        """Encode images to latent space"""
        # For now, use a simple projection since VAE_Encoder expects 3 channels
        # In practice, you might want to adapt this for SAR images
        
        if noise is None:
            # Create noise tensor with appropriate shape for latent space
            # VAE encoder downsamples by factor of 8, outputs 4 channels
            batch_size, _, height, width = images.shape
            latent_height, latent_width = height // 8, width // 8
            noise = torch.randn(batch_size, 4, latent_height, latent_width, device=images.device)
        
        latents = self.vae_encoder(images, noise)
        return latents
    
    def decode_latents(self, latents):
        """Decode latents back to images"""
        images = self.vae_decoder(latents)
        return images
    def forward(self, sar_images, optical_images, prompts, noise=None, timesteps=None):
        """
        Forward pass for training - learns to generate optical images from SAR images
        
        Args:
            sar_images: Input SAR images (B, 3, H, W)
            optical_images: Target optical images (B, 3, H, W)
            prompts: Text prompts (list of strings)
            noise: Noise to add during training (B, 4, H//8, W//8)
            timesteps: Timesteps for diffusion (B,)
        """
        batch_size = sar_images.shape[0]
        device = sar_images.device
        
        # Encode text prompts
        text_embeddings = self.encode_text(prompts)
        
        # Encode SAR images to latent space (condition)
        sar_latents = self.encode_images(sar_images)
        
        # Encode optical images to latent space (target)
        optical_latents = self.encode_images(optical_images)
        
        # For training, we add noise to the target optical latents
        if noise is None:
            noise = torch.randn_like(optical_latents)
        
        if timesteps is None:
            timesteps = torch.randint(0, 1000, (batch_size,), device=device)
            
        # Add noise to optical latents (this is what we want to denoise to)
        noisy_optical_latents = self._add_noise(optical_latents, noise, timesteps)
        
        # Concatenate SAR latents with noisy optical latents as input to diffusion
        # This allows the model to see both the condition (SAR) and noisy target
        diffusion_input = torch.cat([sar_latents, noisy_optical_latents], dim=1)  # (B, 8, H//8, W//8)
        
        # Convert timesteps to embeddings for the diffusion model
        timestep_embeddings = get_timestep_embedding(timesteps, 320)
        
        # Predict noise using the diffusion model
        predicted_noise = self.diffusion(diffusion_input, text_embeddings, timestep_embeddings)
        
        return predicted_noise, noise
    
    def _add_noise(self, latents, noise, timesteps):
        """Add noise to latents according to diffusion schedule"""
        # Simplified noise addition - in practice, you'd use proper DDPM schedule
        alpha = 1.0 - (timesteps.float() / 1000.0).view(-1, 1, 1, 1)
        noisy_latents = alpha * latents + (1 - alpha) * noise
        return noisy_latents
    def generate(self, sar_images, prompts, num_steps=50):
        """Generate colorized images from SAR images and prompts"""
        self.eval()
        
        with torch.no_grad():
            batch_size = sar_images.shape[0]
            device = sar_images.device
            
            # Encode text prompts
            text_embeddings = self.encode_text(prompts)
            
            # Encode SAR images to latent space (this is our condition)
            sar_latents = self.encode_images(sar_images)
            
            # Start with pure noise for the optical image generation
            optical_latents = torch.randn_like(sar_latents)
            
            # Denoising loop
            for step in range(num_steps):
                t = torch.full((batch_size,), step, device=device)
                
                # Convert timesteps to embeddings
                t_embeddings = get_timestep_embedding(t, 320)
                
                # Concatenate SAR latents with current optical latents
                diffusion_input = torch.cat([sar_latents, optical_latents], dim=1)
                
                # Predict noise
                predicted_noise = self.diffusion(diffusion_input, text_embeddings, t_embeddings)
                
                # Remove predicted noise (simplified denoising step)
                alpha = 1.0 - (step / num_steps)
                optical_latents = optical_latents - alpha * predicted_noise
            
            # Decode optical latents to images
            generated_images = self.decode_latents(optical_latents)
            
            return generated_images


class SARTrainer:
    """Trainer for SAR Colorization Model"""
    
    def __init__(
        self,
        config: dict,
        dataset_path: str,
        save_dir: str = "./checkpoints",
        log_dir: str = "./logs",
        clip_checkpoint: str = None
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
        
        # Initialize tokenizer
        self.tokenizer = self._load_tokenizer()
          # Initialize model
        self.model = self._setup_model(clip_checkpoint)
        
        # Setup data loaders  
        self.train_loader, self.val_loader = self._setup_data_loaders()
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
          # Setup loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # Loss weights from config
        loss_config = self.config.get("loss", {})
        self.mse_weight = loss_config.get("mse_loss_weight", 1.0)
        self.l1_weight = loss_config.get("l1_loss_weight", 0.1)
        
        # Setup tensorboard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(os.path.join(log_dir, f"sar_training_{timestamp}"))
          # Training state
        self.current_epoch = 0
        self.global_step = 0
    def _load_tokenizer(self) -> SimpleTokenizer:
        """Load the tokenizer"""
        tokenizer_path = os.path.join(self.save_dir, "tokenizer.json")
        
        if os.path.exists(tokenizer_path):
            print("Loading existing tokenizer...")
            tokenizer = SimpleTokenizer()
            tokenizer.load_vocab(tokenizer_path)
        else:
            print("Creating new tokenizer...")
            from tokenizer import create_tokenizer_from_dataset
            tokenizer = create_tokenizer_from_dataset(
                self.dataset_path,
                vocab_size=self.config.get("clip_config", {}).get("n_vocab", 1000)
            )
            tokenizer.save_vocab(tokenizer_path)
        
        return tokenizer
    
    def _setup_model(self, clip_checkpoint: str = None) -> SARColorizationModel:
        """Setup the complete model"""
        model = SARColorizationModel(
            clip_config=self.config.get("clip_config", {}),
            diffusion_config=self.config.get("diffusion_config", {}),
            tokenizer=self.tokenizer
        )
        
        # Load pre-trained CLIP weights if available
        if clip_checkpoint and os.path.exists(clip_checkpoint):
            print(f"Loading CLIP weights from {clip_checkpoint}")
            clip_state = torch.load(clip_checkpoint, map_location=self.device)
            model.clip.load_state_dict(clip_state['model_state_dict'])
        
        model = model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        return model
    def _setup_data_loaders(self) -> tuple[DataLoader, DataLoader]:
        """Setup data loaders"""
        train_loader, val_loader = create_dataloaders(
            dataset_path=self.dataset_path,
            batch_size=self.config.get("training", {}).get("batch_size", 4),
            train_split=self.config.get("data", {}).get("train_split", 0.8),
            num_workers=self.config.get("data", {}).get("num_workers", 4),
            seed=self.config.get("data", {}).get("seed", 42)
        )
        
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        return train_loader, val_loader
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer"""
        training_config = self.config.get("training", {})
        optimizer_name = training_config.get("optimizer", "adamw")
        
        if optimizer_name == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=training_config.get("learning_rate", 1e-4),
                weight_decay=training_config.get("weight_decay", 0.01)
            )
        elif optimizer_name == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=training_config.get("learning_rate", 1e-4),
                weight_decay=training_config.get("weight_decay", 0.01)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        training_config = self.config.get("training", {})
        scheduler_name = training_config.get("scheduler", "cosine")
        
        if scheduler_name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config.get("epochs", 100)
            )
        elif scheduler_name == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=training_config.get("step_size", 30),
                gamma=training_config.get("gamma", 0.1)
            )
        else:
            return None
    def train_epoch(self) -> dict:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Training Epoch {self.current_epoch}",
            unit="batch",
            leave=True,
            dynamic_ncols=True,
            ascii=True
        )
        
        for batch in progress_bar:
            # Prepare data
            sar_images = batch['s1_image'].to(self.device)
            optical_images = batch['s2_image'].to(self.device)
            prompts = batch['prompt']
              # Forward pass
            self.optimizer.zero_grad()
            predicted_noise, target_noise = self.model(sar_images, optical_images, prompts)
            
            # Compute loss (combination of MSE and L1 for better training)
            mse_loss = self.mse_loss(predicted_noise, target_noise)
            l1_loss = self.l1_loss(predicted_noise, target_noise)
            loss = self.mse_weight * mse_loss + self.l1_weight * l1_loss
            
            # Backward pass
            loss.backward()
              # Gradient clipping
            max_grad_norm = self.config.get("training", {}).get("max_grad_norm", 0)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_grad_norm
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
              # Log to tensorboard
            log_interval = self.config.get("training", {}).get("log_interval", 100)
            if self.global_step % log_interval == 0:
                self.writer.add_scalar("Loss/Train", loss.item(), self.global_step)
                self.writer.add_scalar("Learning_Rate", self.optimizer.param_groups[0]['lr'], self.global_step)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/num_batches:.4f}"
            })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"train_loss": avg_loss}
    def validate(self) -> dict:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(
                self.val_loader, 
                desc="Validation",
                unit="batch",
                leave=True,
                dynamic_ncols=True,
                ascii=True
            )
            
            for batch in progress_bar:
                # Prepare data
                sar_images = batch['s1_image'].to(self.device)
                optical_images = batch['s2_image'].to(self.device)
                prompts = batch['prompt']
                # Forward pass                
                predicted_noise, target_noise = self.model(sar_images, optical_images, prompts)
                # Compute loss (combination of MSE and L1 for better training)
                mse_loss = self.mse_loss(predicted_noise, target_noise)
                l1_loss = self.l1_loss(predicted_noise, target_noise)
                loss = self.mse_weight * mse_loss + self.l1_weight * l1_loss
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'val_loss': f"{loss.item():.4f}",
                    'avg_loss': f"{total_loss/num_batches:.4f}"
                })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"val_loss": avg_loss}
    
    def save_checkpoint(self, metrics: dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'metrics': metrics
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.save_dir, f"sar_model_epoch_{self.current_epoch}.pt")
        # torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.save_dir, "sar_model_best.pt")
            torch.save(checkpoint, best_path)
            print(f"New best model saved: {best_path}")
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch}")
    def train(self, resume_from: str = None):
        """Main training loop"""
        if resume_from:
            self.load_checkpoint(resume_from)
        
        best_val_loss = float('inf')
        total_epochs = self.config.get("training", {}).get("epochs", 100)
        
        # Create epoch progress bar
        epoch_progress = tqdm(
            range(self.current_epoch, total_epochs),
            desc="Training Progress",
            unit="epoch",
            position=0,
            leave=True,
            dynamic_ncols=True,
            ascii=True
        )
        
        for epoch in epoch_progress:
            self.current_epoch = epoch            
            print(f"\nEpoch {epoch}/{total_epochs}")
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
            print(f"Val Loss: {metrics['val_loss']:.4f}")
            
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
            if epoch % self.config.get("training", {}).get("save_every", 10) == 0 or is_best:
                self.save_checkpoint(metrics, is_best)
        
        print("\nTraining completed!")
        self.writer.close()


def get_default_config():
    """Get default training configuration"""
    return {
        "clip_config": {
            "n_vocab": 1000,
            "n_embd": 768,
            "max_length": 77,
            "n_head": 12,
            "n_layers": 12
        },        
        "diffusion_config": {
            "in_channels": 8,  # 4 (SAR latents) + 4 (optical latents)
            "out_channels": 4,  # Predicting noise in 4-channel latent space
            "model_channels": 320,
            "attention_resolutions": [4, 2, 1],
            "num_res_blocks": 2,
            "channel_mult": [1, 2, 4, 4],
            "num_heads": 8,
            "use_spatial_transformer": True,
            "context_dim": 768
        },
        "training": {
            "batch_size": 4,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "epochs": 100,
            "warmup_steps": 2000,
            "save_every": 10,
            "validate_every": 5,
            "mixed_precision": True,
            "gradient_accumulation_steps": 4,
            "max_grad_norm": 1.0,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "log_interval": 100
        },
        "data": {
            "train_split": 0.8,
            "num_workers": 4,
            "pin_memory": True,
            "seed": 42
        },        "loss": {
            "mse_loss_weight": 1.0,
            "l1_loss_weight": 0.1,
            "perceptual_loss_weight": 0.1,
            "adversarial_loss_weight": 0.01
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Train SAR Colorization Model")
    parser.add_argument("--dataset_path", type=str, default="../Dataset",
                       help="Path to the dataset folder")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config JSON file")
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs",
                       help="Directory for tensorboard logs")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--clip_checkpoint", type=str, default=None,
                       help="Path to pre-trained CLIP checkpoint")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_config()
    
    # Save configuration
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Configuration:")
    for section, params in config.items():
        print(f"{section}:")
        for key, value in params.items():
            print(f"  {key}: {value}")
    
    # Initialize trainer
    trainer = SARTrainer(
        config=config,
        dataset_path=args.dataset_path,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        clip_checkpoint=args.clip_checkpoint
    )
    
    # Start training
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
