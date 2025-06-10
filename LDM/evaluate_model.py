#!/usr/bin/env python3
"""
SAR to Optical Image Translation Evaluation Script

This script evaluates the trained model on the test dataset and computes various metrics:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
- FID (Fréchet Inception Distance)
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
"""

import yaml
import argparse
import torch
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import json
from datetime import datetime

# Model imports
from models.unet_cond_base import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils.text_utils import get_tokenizer_and_model, get_text_representation
from dataset.sar_dataset import SARDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ImageMetrics:
    """Class to compute various image quality metrics"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.lpips_net = None
        
    def setup_lpips(self):
        """Initialize LPIPS network"""
        try:
            import lpips
            self.lpips_net = lpips.LPIPS(net='alex').to(self.device)
            print("✓ LPIPS network loaded successfully")
        except ImportError:
            print("⚠ LPIPS not available. Install with: pip install lpips")
            self.lpips_net = None
    
    def psnr(self, img1, img2, data_range=1.0):
        """Compute PSNR between two images"""
        img1_np = img1.cpu().numpy()
        img2_np = img2.cpu().numpy()
        
        # Convert from CHW to HWC for skimage
        if len(img1_np.shape) == 4:  # Batch dimension
            psnr_values = []
            for i in range(img1_np.shape[0]):
                img1_hwc = np.transpose(img1_np[i], (1, 2, 0))
                img2_hwc = np.transpose(img2_np[i], (1, 2, 0))
                psnr_val = peak_signal_noise_ratio(img1_hwc, img2_hwc, data_range=data_range)
                psnr_values.append(psnr_val)
            return np.mean(psnr_values)
        else:
            img1_hwc = np.transpose(img1_np, (1, 2, 0))
            img2_hwc = np.transpose(img2_np, (1, 2, 0))
            return peak_signal_noise_ratio(img1_hwc, img2_hwc, data_range=data_range)
    
    def ssim(self, img1, img2, data_range=1.0):
        """Compute SSIM between two images"""
        img1_np = img1.cpu().numpy()
        img2_np = img2.cpu().numpy()
        
        if len(img1_np.shape) == 4:  # Batch dimension
            ssim_values = []
            for i in range(img1_np.shape[0]):
                img1_hwc = np.transpose(img1_np[i], (1, 2, 0))
                img2_hwc = np.transpose(img2_np[i], (1, 2, 0))
                ssim_val = structural_similarity(img1_hwc, img2_hwc, 
                                               data_range=data_range, 
                                               channel_axis=2, 
                                               multichannel=True)
                ssim_values.append(ssim_val)
            return np.mean(ssim_values)
        else:
            img1_hwc = np.transpose(img1_np, (1, 2, 0))
            img2_hwc = np.transpose(img2_np, (1, 2, 0))
            return structural_similarity(img1_hwc, img2_hwc, 
                                       data_range=data_range, 
                                       channel_axis=2, 
                                       multichannel=True)
    
    def lpips(self, img1, img2):
        """Compute LPIPS between two images"""
        if self.lpips_net is None:
            return None
        
        # Ensure images are in [-1, 1] range for LPIPS
        img1_norm = img1 * 2.0 - 1.0 if img1.max() <= 1.0 else img1
        img2_norm = img2 * 2.0 - 1.0 if img2.max() <= 1.0 else img2
        
        with torch.no_grad():
            lpips_val = self.lpips_net(img1_norm, img2_norm)
        return lpips_val.mean().item()
    
    def mse(self, img1, img2):
        """Compute MSE between two images"""
        return F.mse_loss(img1, img2).item()
    
    def mae(self, img1, img2):
        """Compute MAE between two images"""
        return F.l1_loss(img1, img2).item()


def load_models(config):
    """Load trained VQVAE and LDM models"""
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
        print(f"✓ Loaded VQVAE from {vqvae_path}")
    else:
        raise FileNotFoundError(f"VQVAE checkpoint not found at {vqvae_path}")
    
    vqvae.eval()
    
    # Load Diffusion Model
    ldm = Unet(im_channels=autoencoder_model_config['z_channels'],
               model_config=diffusion_model_config).to(device)
    
    ldm_path = os.path.join(train_config['task_name'], train_config['ldm_ckpt_name'])
    if os.path.exists(ldm_path):
        ldm.load_state_dict(torch.load(ldm_path, map_location=device))
        print(f"✓ Loaded LDM from {ldm_path}")
    else:
        raise FileNotFoundError(f"LDM checkpoint not found at {ldm_path}")
    
    ldm.eval()
    
    # Load text models
    condition_config = diffusion_model_config['condition_config']
    text_tokenizer, text_model = get_tokenizer_and_model(
        condition_config['text_condition_config']['text_embed_model'], 
        device=device)
    
    return vqvae, ldm, text_tokenizer, text_model


def preprocess_sar_image(sar_tensor, device):
    """Preprocess SAR image tensor for conditioning"""
    # Move to device first
    sar_tensor = sar_tensor.to(device)
    
    # Ensure the SAR image is in [-1, 1] range
    if sar_tensor.max() <= 1.0 and sar_tensor.min() >= 0.0:
        # Convert from [0, 1] to [-1, 1]
        sar_tensor = sar_tensor * 2.0 - 1.0
    return sar_tensor


def sample_with_conditioning(ldm, scheduler, sar_image, text_prompt, text_tokenizer, text_model, 
                           diffusion_config, autoencoder_config, condition_config, guidance_scale=7.5):
    """Sample optical image conditioned on SAR image and text prompt"""
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
    for i in reversed(range(diffusion_config['num_timesteps'])):
        t = torch.full((1,), i, dtype=torch.long).to(device)
        
        with torch.no_grad():
            # Conditional prediction
            noise_pred_cond = ldm(xt, t, cond_input=cond_input)
            
            # Unconditional prediction for classifier-free guidance
            noise_pred_uncond = ldm(xt, t, cond_input=uncond_input)
            
            # Apply classifier-free guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            # Denoise step
            xt, x0 = scheduler.sample_prev_timestep(xt, noise_pred, i)
    
    return xt


def evaluate_model(config, num_samples=None, guidance_scale=7.5, save_samples=True):
    """
    Evaluate the model on test dataset
    
    Args:
        config: Configuration dictionary
        num_samples: Number of samples to evaluate (None for all)
        guidance_scale: Classifier-free guidance scale
        save_samples: Whether to save sample images
    """
    # Load models
    vqvae, ldm, text_tokenizer, text_model = load_models(config)
    
    # Create scheduler
    diffusion_config = config['diffusion_params']
    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config['num_timesteps'],
        beta_start=diffusion_config['beta_start'],
        beta_end=diffusion_config['beta_end']
    )
      # Load test dataset
    dataset_config = config['dataset_params']
    condition_config = config['ldm_params']['condition_config']
    
    test_dataset = SARDataset(
        split='test',
        im_path=dataset_config['im_path'],
        im_size=dataset_config['im_size'],
        im_channels=dataset_config['im_channels'],
        condition_config=condition_config,
        train_split=dataset_config['train_split'],
        val_split=dataset_config['val_split'],
        test_split=dataset_config['test_split'],
        use_latents=False  # Disable latents for evaluation - we need original images
    )
    
    print(f"Test dataset loaded: {len(test_dataset)} samples")
    
    # Limit samples if specified
    if num_samples and num_samples < len(test_dataset):
        indices = torch.randperm(len(test_dataset))[:num_samples]
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
        print(f"Evaluating on {num_samples} random samples")
    
    # Create data loader
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    
    # Initialize metrics
    metrics_calculator = ImageMetrics(device)
    metrics_calculator.setup_lpips()
    
    # Storage for results
    results = {
        'psnr': [],
        'ssim': [],
        'lpips': [],
        'mse': [],
        'mae': [],
        'sample_info': []
    }
    
    # Create results directory
    results_dir = os.path.join(config['train_params']['task_name'], 'evaluation_results')
    os.makedirs(results_dir, exist_ok=True)
    
    if save_samples:
        samples_dir = os.path.join(results_dir, 'sample_images')
        os.makedirs(samples_dir, exist_ok=True)
    
    print("Starting evaluation...")
    
    # Evaluation loop
    for i, data in enumerate(tqdm(test_loader, desc="Evaluating")):
        try:            # Unpack data
            if isinstance(data, tuple):
                ground_truth, cond_inputs = data
                
                # Handle text prompt (might be a list from DataLoader batching)
                if 'text' in cond_inputs:
                    text_prompt = cond_inputs['text']
                    if isinstance(text_prompt, list):
                        text_prompt = text_prompt[0]  # Take first item from batch
                else:
                    text_prompt = "Colorise image"
                
                sar_image = cond_inputs['image'] if 'image' in cond_inputs else None
            else:
                ground_truth = data
                text_prompt = "Colorise image"
                sar_image = None
            
            ground_truth = ground_truth.to(device)
            
            if sar_image is None:
                print(f"Warning: No SAR image for sample {i}, skipping...")
                continue
            
            # Handle case where sar_image might be a list (from DataLoader batching)
            if isinstance(sar_image, list):
                sar_image = sar_image[0]  # Take first (and only) item from batch
            
            # Ensure sar_image is a tensor
            if not isinstance(sar_image, torch.Tensor):
                print(f"Warning: SAR image is not a tensor, got {type(sar_image)}, skipping...")
                continue
            
            sar_image = preprocess_sar_image(sar_image, device)
            
            # Generate optical image
            with torch.no_grad():
                latent = sample_with_conditioning(
                    ldm, scheduler, sar_image, text_prompt, text_tokenizer, text_model,
                    diffusion_config, config['autoencoder_params'], 
                    config['ldm_params']['condition_config'], guidance_scale
                )
                
                # Decode latent to image
                generated_image = vqvae.decode(latent)
            
            # Post-process images
            generated_image = (generated_image + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
            generated_image = torch.clamp(generated_image, 0, 1)
            
            ground_truth_norm = (ground_truth + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
            ground_truth_norm = torch.clamp(ground_truth_norm, 0, 1)
            
            # Compute metrics
            psnr_val = metrics_calculator.psnr(generated_image, ground_truth_norm)
            ssim_val = metrics_calculator.ssim(generated_image, ground_truth_norm)
            lpips_val = metrics_calculator.lpips(generated_image, ground_truth_norm)
            mse_val = metrics_calculator.mse(generated_image, ground_truth_norm)
            mae_val = metrics_calculator.mae(generated_image, ground_truth_norm)
            
            # Store results
            results['psnr'].append(psnr_val)
            results['ssim'].append(ssim_val)
            if lpips_val is not None:
                results['lpips'].append(lpips_val)
            results['mse'].append(mse_val)
            results['mae'].append(mae_val)
            results['sample_info'].append({
                'sample_id': i,
                'text_prompt': text_prompt,
                'psnr': psnr_val,
                'ssim': ssim_val,
                'lpips': lpips_val,
                'mse': mse_val,
                'mae': mae_val
            })
            
            # Save sample images
            if save_samples and i < 10:  # Save first 10 samples
                from torchvision.utils import save_image
                
                # Denormalize SAR image for saving
                sar_image_norm = (sar_image + 1) / 2
                sar_image_norm = torch.clamp(sar_image_norm, 0, 1)
                
                # Create comparison grid
                comparison = torch.cat([sar_image_norm, generated_image, ground_truth_norm], dim=0)
                save_image(comparison, 
                          os.path.join(samples_dir, f'sample_{i:03d}_comparison.png'),
                          nrow=3, normalize=False)
            
        except Exception as e:
            print(f"Error processing sample {i}: {str(e)}")
            continue
    
    return results, results_dir


def generate_report(results, results_dir, config):
    """Generate comprehensive evaluation report"""
    
    # Calculate summary statistics
    summary_stats = {}
    for metric in ['psnr', 'ssim', 'lpips', 'mse', 'mae']:
        if results[metric]:  # Check if metric has values
            values = np.array(results[metric])
            summary_stats[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
    
    # Create detailed report
    report = {
        'evaluation_info': {
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(results['sample_info']),
            'model_config': config['train_params']['task_name'],
            'guidance_scale': config['train_params'].get('cf_guidance_scale', 7.5)
        },
        'summary_statistics': summary_stats,
        'detailed_results': results['sample_info']
    }
    
    # Save JSON report
    report_path = os.path.join(results_dir, 'evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create plots
    create_metric_plots(results, results_dir)
    
    # Print summary to console
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Evaluated {len(results['sample_info'])} samples")
    print(f"Results saved in: {results_dir}")
    print("\nMetric Summary:")
    print("-" * 40)
    
    for metric, stats in summary_stats.items():
        print(f"{metric.upper():>6}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"       Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"       Median: {stats['median']:.4f}")
        print()
    
    print(f"Detailed report: {report_path}")
    print("="*60)
    
    return report_path


def create_metric_plots(results, results_dir):
    """Create visualization plots for metrics"""
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    
    # Create subplots for all metrics
    metrics_to_plot = ['psnr', 'ssim', 'mse', 'mae']
    if results['lpips']:
        metrics_to_plot.append('lpips')
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_to_plot):
        if i < len(axes) and results[metric]:
            ax = axes[i]
            values = results[metric]
            
            # Histogram
            ax.hist(values, bins=20, alpha=0.7, edgecolor='black')
            ax.set_title(f'{metric.upper()} Distribution')
            ax.set_xlabel(metric.upper())
            ax.set_ylabel('Frequency')
            ax.axvline(np.mean(values), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(values):.4f}')
            ax.legend()
    
    # Remove empty subplots
    for i in range(len(metrics_to_plot), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'metrics_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create time series plot (sample-wise metrics)
    if len(results['sample_info']) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        sample_indices = range(len(results['sample_info']))
        
        plot_metrics = ['psnr', 'ssim', 'mse', 'mae']
        for i, metric in enumerate(plot_metrics):
            if i < len(axes):
                ax = axes[i]
                values = [sample[metric] for sample in results['sample_info'] if sample[metric] is not None]
                ax.plot(sample_indices[:len(values)], values, marker='o', markersize=2)
                ax.set_title(f'{metric.upper()} per Sample')
                ax.set_xlabel('Sample Index')
                ax.set_ylabel(metric.upper())
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'metrics_per_sample.png'), dpi=300, bbox_inches='tight')
        plt.close()


def custom_collate_fn(batch):
    """Custom collate function to handle SAR dataset batching"""
    if len(batch) == 1:
        # Single sample case
        sample = batch[0]
        if isinstance(sample, tuple):
            # (target_image, cond_inputs) format
            target, cond_inputs = sample
            
            # Convert target to batch format
            target_batch = target.unsqueeze(0)
            
            # Handle conditioning inputs
            cond_batch = {}
            for key, value in cond_inputs.items():
                if key == 'text':
                    # Text should remain as list for batch
                    cond_batch[key] = [value] if isinstance(value, str) else value
                elif key == 'image':
                    # Image should be batched
                    cond_batch[key] = value.unsqueeze(0) if isinstance(value, torch.Tensor) else value
                else:
                    cond_batch[key] = value
            
            return target_batch, cond_batch
        else:
            # Just target image
            return sample.unsqueeze(0)
    else:
        # Multiple samples - use default collation
        return torch.utils.data.dataloader.default_collate(batch)


def main():
    parser = argparse.ArgumentParser(description='Evaluate SAR to Optical Image Translation Model')
    parser.add_argument('--config', default='config/sar_config.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to evaluate (default: all)')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                       help='Classifier-free guidance scale')
    parser.add_argument('--no_save_samples', action='store_true',
                       help='Do not save sample images')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    print("="*60)
    print("SAR TO OPTICAL IMAGE TRANSLATION EVALUATION")
    print("="*60)
    print(f"Configuration: {args.config}")
    print(f"Device: {device}")
    print(f"Guidance Scale: {args.guidance_scale}")
    print(f"Number of samples: {args.num_samples or 'All'}")
    print("="*60)
    
    try:
        # Run evaluation
        results, results_dir = evaluate_model(
            config, 
            num_samples=args.num_samples,
            guidance_scale=args.guidance_scale,
            save_samples=not args.no_save_samples
        )
        
        # Generate report
        report_path = generate_report(results, results_dir, config)
        
        print(f"\n✓ Evaluation completed successfully!")
        print(f"✓ Results saved in: {results_dir}")
        
    except Exception as e:
        print(f"\n✗ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
