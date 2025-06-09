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

from dataset import SARDataset, create_dataloaders
from tokenizer import SimpleTokenizer, create_tokenizer_from_dataset
from clip import CLIP


class CLIPTrainer:
    """Trainer for CLIP model on SAR colorization prompts"""
    
    def __init__(
        self,
        config: dict,
        dataset_path: str,
        save_dir: str = "./checkpoints",
        log_dir: str = "./logs"
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
        self.tokenizer = self._setup_tokenizer()
        
        # Initialize model
        self.model = self._setup_model()
        
        # Setup data loaders
        self.train_loader, self.val_loader = self._setup_data_loaders()
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup tensorboard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(os.path.join(log_dir, f"clip_training_{timestamp}"))
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        
    def _setup_tokenizer(self) -> SimpleTokenizer:
        """Setup or load tokenizer"""
        tokenizer_path = os.path.join(self.save_dir, "tokenizer.json")
        
        if os.path.exists(tokenizer_path):
            print("Loading existing tokenizer...")
            tokenizer = SimpleTokenizer()
            tokenizer.load_vocab(tokenizer_path)
        else:
            print("Creating new tokenizer from dataset...")
            tokenizer = create_tokenizer_from_dataset(
                self.dataset_path,
                vocab_size=self.config["vocab_size"]
            )
            tokenizer.save_vocab(tokenizer_path)
            print(f"Tokenizer saved to {tokenizer_path}")
        
        return tokenizer
    
    def _setup_model(self) -> CLIP:
        """Setup CLIP model"""
        model = CLIP(
            n_vocab=self.tokenizer.get_vocab_size(),
            n_embd=self.config["n_embd"],
            n_token=self.config["max_length"],
            n_head=self.config["n_head"],
            n_layers=self.config["n_layers"]
        )
        
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
            batch_size=self.config["batch_size"],
            train_split=self.config["train_split"],
            num_workers=self.config["num_workers"],
            seed=self.config["seed"]
        )
        
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        return train_loader, val_loader
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer"""
        if self.config["optimizer"] == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"]
            )
        elif self.config["optimizer"] == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"]
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config['optimizer']}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        if self.config["scheduler"] == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config["epochs"]
            )
        elif self.config["scheduler"] == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config["step_size"],
                gamma=self.config["gamma"]
            )
        else:
            return None
    
    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss for CLIP training"""
        # Simple masked language modeling loss for now
        # In a full CLIP implementation, this would be contrastive loss
        
        # Mask padding tokens
        mask = (targets != self.tokenizer.get_pad_token_id())
        
        # Reshape for loss computation
        outputs = outputs.view(-1, outputs.size(-1))  # (batch_size * seq_len, vocab_size)
        targets = targets.view(-1)  # (batch_size * seq_len)
        mask = mask.view(-1)  # (batch_size * seq_len)
        
        # Only compute loss on non-padded tokens
        if mask.sum() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Cross entropy loss
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        losses = loss_fn(outputs, targets)
        
        # Apply mask and compute mean
        masked_losses = losses * mask.float()
        loss = masked_losses.sum() / mask.sum()
        
        return loss
    def train_epoch(self) -> dict:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {self.current_epoch}",
            unit="batch",
            leave=True,
            dynamic_ncols=True,
            ascii=True
        )
        
        for batch in progress_bar:
            # Prepare data
            prompts = batch['prompt']
            token_ids = self.tokenizer.encode_batch(prompts).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(token_ids)
            
            # For now, we'll use a simple next-token prediction loss
            # In practice, you'd want to implement proper contrastive learning
            targets = token_ids.clone()
            loss = self.compute_loss(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get("grad_clip", 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["grad_clip"]
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Log to tensorboard
            if self.global_step % self.config["log_interval"] == 0:
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
                prompts = batch['prompt']
                token_ids = self.tokenizer.encode_batch(prompts).to(self.device)
                outputs = self.model(token_ids)
                targets = token_ids.clone()
                loss = self.compute_loss(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'val_loss': f"{loss.item():.4f}",
                    'avg_loss': f"{total_loss/num_batches:.4f}"
                })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'val_loss': avg_loss,
            'perplexity': np.exp(avg_loss)
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'tokenizer_vocab': self.tokenizer.vocab
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(self.save_dir, "latest_clip.pt")
        torch.save(checkpoint, latest_path)
        
        # Save epoch checkpoint
        epoch_path = os.path.join(self.save_dir, f"clip_epoch_{self.current_epoch}.pt")
        torch.save(checkpoint, epoch_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.save_dir, "best_clip.pt")
            torch.save(checkpoint, best_path)
            print(f"New best model saved to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config['epochs']} epochs")
        
        best_val_loss = float('inf')
        
        # Create epoch progress bar
        epoch_progress = tqdm(
            range(self.current_epoch, self.config['epochs']),
            desc="Training Progress",
            unit="epoch",
            position=0,
            leave=True,
            dynamic_ncols=True,
            ascii=True
        )
        
        for epoch in epoch_progress:
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Log metrics
            self.writer.add_scalar('Loss/Train', train_metrics['train_loss'], epoch)
            self.writer.add_scalar('Loss/Val', val_metrics['val_loss'], epoch)
            self.writer.add_scalar('Perplexity/Val', val_metrics['perplexity'], epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            print(f"Epoch {epoch}: Train Loss: {train_metrics['train_loss']:.4f}, "
                  f"Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"Perplexity: {val_metrics['perplexity']:.4f}")
            
            # Update epoch progress bar
            epoch_progress.set_postfix({
                'train_loss': f"{train_metrics['train_loss']:.4f}",
                'val_loss': f"{val_metrics['val_loss']:.4f}",
                'perplexity': f"{val_metrics['perplexity']:.4f}"
            })
            
            # Save checkpoint
            is_best = val_metrics['val_loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['val_loss']
            
            if (epoch + 1) % self.config['save_every'] == 0:
                self.save_checkpoint(is_best)
            
            # Step scheduler
            if self.scheduler:
                self.scheduler.step()
        
        # Save final checkpoint
        self.save_checkpoint()
        self.writer.close()
        
        print("Training completed!")


def get_default_config():
    """Get default training configuration"""
    return {
        # Model parameters
        "vocab_size": 1000,
        "n_embd": 768,
        "max_length": 77,
        "n_head": 12,
        "n_layers": 12,
        
        # Training parameters
        "batch_size": 8,
        "epochs": 100,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "grad_clip": 1.0,
        
        # Data parameters
        "train_split": 0.8,
        "num_workers": 4,
        "seed": 42,
        
        # Logging parameters
        "log_interval": 100,
        "save_interval": 10,
        
        # Scheduler parameters (for step scheduler)
        "step_size": 30,
        "gamma": 0.1
    }


def main():
    parser = argparse.ArgumentParser(description="Train CLIP model for SAR colorization")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to the dataset folder")
    parser.add_argument("--config_file", type=str, default="configs/clip_config.json",
                       help="Path to configuration file")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/clip",
                       help="Directory to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs/clip",
                       help="Directory for tensorboard logs")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    # Override config options
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--learning_rate", type=float, help="Override learning rate")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--vocab_size", type=int, help="Override vocabulary size")
    parser.add_argument("--embedding_dim", type=int, help="Override embedding dimension")
    parser.add_argument("--max_length", type=int, help="Override max sequence length")
    parser.add_argument("--num_heads", type=int, help="Override number of attention heads")
    parser.add_argument("--num_layers", type=int, help="Override number of layers")
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            "model": {
                "vocab_size": 1000,
                "embedding_dim": 768,
                "max_length": 77,
                "num_heads": 12,
                "num_layers": 12
            },
            "training": {
                "batch_size": 32,
                "learning_rate": 1e-4,
                "weight_decay": 0.01,
                "epochs": 50,
                "warmup_steps": 1000,
                "save_every": 5,
                "optimizer": "adamw",
                "scheduler": "cosine",
                "grad_clip": 1.0
            },
            "data": {
                "train_split": 0.8,
                "num_workers": 4,
                "pin_memory": True,
                "seed": 42
            }
        }
    
    # Override with command line arguments
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.vocab_size is not None:
        config["model"]["vocab_size"] = args.vocab_size
    if args.embedding_dim is not None:
        config["model"]["embedding_dim"] = args.embedding_dim
    if args.max_length is not None:
        config["model"]["max_length"] = args.max_length
    if args.num_heads is not None:
        config["model"]["num_heads"] = args.num_heads
    if args.num_layers is not None:
        config["model"]["num_layers"] = args.num_layers
    
    # Flatten config for trainer
    trainer_config = {
        "vocab_size": config["model"]["vocab_size"],
        "n_embd": config["model"]["embedding_dim"],
        "max_length": config["model"]["max_length"],
        "n_head": config["model"]["num_heads"],
        "n_layers": config["model"]["num_layers"],
        **config["training"],
        **config["data"]
    }
    
    print("Starting CLIP training with configuration:")
    print(json.dumps(config, indent=2))
    
    # Initialize trainer
    trainer = CLIPTrainer(
        config=trainer_config,
        dataset_path=args.dataset_path,
        save_dir=args.save_dir,
        log_dir=args.log_dir
    )
    
    # Resume from checkpoint if specified
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
        print(f"Resumed training from {args.resume_from}")
    
    # Start training
    trainer.train()
    
    print("Training completed!")


if __name__ == "__main__":
    main()
