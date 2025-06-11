"""
Loss Plotting Utility for SAR Image Colorization Training

This script provides functions to plot training loss curves from the saved loss data.
Works with both autoencoder and LDM training loss histories.

Usage:
    python plot_losses.py --task_name sar_colorization --type both
    python plot_losses.py --task_name sar_colorization --type autoencoder
    python plot_losses.py --task_name sar_colorization --type ldm
"""

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_loss_data(task_name, loss_type):
    """Load loss data from JSON file"""
    losses_dir = os.path.join(task_name, 'losses')
    json_path = os.path.join(losses_dir, f'{loss_type}_loss_history.json')
    
    if not os.path.exists(json_path):
        print(f"Error: Loss data not found at {json_path}")
        return None
    
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_autoencoder_losses(loss_data, save_path=None):
    """Plot autoencoder loss curves"""
    # Extract epoch-level data
    epochs = []
    avg_losses = []
    best_losses = []
    
    # Extract batch-level data
    batch_data = []
    
    for epoch_data in loss_data['epochs']:
        epochs.append(epoch_data['epoch'])
        avg_losses.append(epoch_data['avg_loss'])
        best_losses.append(epoch_data['best_loss'])
        
        for batch_data_point in epoch_data['batches']:
            batch_data.append({
                'epoch': epoch_data['epoch'],
                'batch': batch_data_point['batch'],
                'recon_loss': batch_data_point['recon_loss'],
                'codebook_loss': batch_data_point['codebook_loss'],
                'commitment_loss': batch_data_point['commitment_loss'],
                'total_loss': batch_data_point['total_loss']
            })
    
    # Create DataFrame for easier plotting
    df_batches = pd.DataFrame(batch_data)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Autoencoder Training Loss Curves', fontsize=16, fontweight='bold')
    
    # 1. Epoch-level average losses
    axes[0, 0].plot(epochs, avg_losses, 'b-', linewidth=2, label='Average Loss')
    axes[0, 0].plot(epochs, best_losses, 'r--', linewidth=2, label='Best Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Epoch-level Loss Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Batch-level total loss over time
    axes[0, 1].plot(range(len(df_batches)), df_batches['total_loss'], 'g-', alpha=0.7, linewidth=1)
    axes[0, 1].set_xlabel('Batch Number')
    axes[0, 1].set_ylabel('Total Loss')
    axes[0, 1].set_title('Batch-level Total Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Loss components breakdown
    axes[1, 0].plot(range(len(df_batches)), df_batches['recon_loss'], label='Reconstruction', alpha=0.8)
    axes[1, 0].plot(range(len(df_batches)), df_batches['codebook_loss'], label='Codebook', alpha=0.8)
    axes[1, 0].plot(range(len(df_batches)), df_batches['commitment_loss'], label='Commitment', alpha=0.8)
    axes[1, 0].set_xlabel('Batch Number')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Loss Components Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Loss distribution by epoch (boxplot)
    if len(epochs) > 1:
        epoch_groups = df_batches.groupby('epoch')['total_loss'].apply(list).reset_index()
        axes[1, 1].boxplot([losses for losses in epoch_groups['total_loss']], 
                          labels=[f'E{e}' for e in epoch_groups['epoch']])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Total Loss Distribution')
        axes[1, 1].set_title('Loss Distribution by Epoch')
        axes[1, 1].tick_params(axis='x', rotation=45)
    else:
        axes[1, 1].hist(df_batches['total_loss'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Total Loss')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Loss Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Autoencoder loss plot saved to {save_path}")
    
    plt.show()


def plot_ldm_losses(loss_data, save_path=None):
    """Plot LDM loss curves"""
    # Extract epoch-level data
    epochs = []
    avg_losses = []
    best_losses = []
    
    # Extract batch-level data
    batch_data = []
    
    for epoch_data in loss_data['epochs']:
        epochs.append(epoch_data['epoch'])
        avg_losses.append(epoch_data['avg_loss'])
        best_losses.append(epoch_data['best_loss'])
        
        for batch_data_point in epoch_data['batches']:
            batch_data.append({
                'epoch': epoch_data['epoch'],
                'batch': batch_data_point['batch'],
                'loss': batch_data_point['loss']
            })
    
    # Create DataFrame for easier plotting
    df_batches = pd.DataFrame(batch_data)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('LDM Training Loss Curves', fontsize=16, fontweight='bold')
    
    # 1. Epoch-level average losses
    axes[0, 0].plot(epochs, avg_losses, 'b-', linewidth=2, label='Average Loss')
    axes[0, 0].plot(epochs, best_losses, 'r--', linewidth=2, label='Best Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Epoch-level Loss Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Batch-level loss over time
    axes[0, 1].plot(range(len(df_batches)), df_batches['loss'], 'g-', alpha=0.7, linewidth=1)
    axes[0, 1].set_xlabel('Batch Number')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Batch-level Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Moving average of batch losses
    window_size = min(50, len(df_batches) // 10)  # Adaptive window size
    if window_size > 1:
        moving_avg = df_batches['loss'].rolling(window=window_size).mean()
        axes[1, 0].plot(range(len(df_batches)), df_batches['loss'], alpha=0.3, color='gray', label='Raw Loss')
        axes[1, 0].plot(range(len(df_batches)), moving_avg, 'r-', linewidth=2, label=f'Moving Avg (window={window_size})')
        axes[1, 0].set_xlabel('Batch Number')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Loss with Moving Average')
        axes[1, 0].legend()
    else:
        axes[1, 0].plot(range(len(df_batches)), df_batches['loss'], 'g-', linewidth=1)
        axes[1, 0].set_xlabel('Batch Number')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Batch-level Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Loss distribution by epoch (boxplot)
    if len(epochs) > 1:
        epoch_groups = df_batches.groupby('epoch')['loss'].apply(list).reset_index()
        axes[1, 1].boxplot([losses for losses in epoch_groups['loss']], 
                          labels=[f'E{e}' for e in epoch_groups['epoch']])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss Distribution')
        axes[1, 1].set_title('Loss Distribution by Epoch')
        axes[1, 1].tick_params(axis='x', rotation=45)
    else:
        axes[1, 1].hist(df_batches['loss'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Loss')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Loss Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"LDM loss plot saved to {save_path}")
    
    plt.show()


def plot_comparison(autoencoder_data, ldm_data, save_path=None):
    """Plot comparison between autoencoder and LDM losses"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Training Loss Comparison: Autoencoder vs LDM', fontsize=16, fontweight='bold')
    
    # Autoencoder epoch losses
    if autoencoder_data:
        ae_epochs = [ep['epoch'] for ep in autoencoder_data['epochs']]
        ae_losses = [ep['avg_loss'] for ep in autoencoder_data['epochs']]
        axes[0].plot(ae_epochs, ae_losses, 'b-', linewidth=2, label='Autoencoder')
    
    # LDM epoch losses
    if ldm_data:
        ldm_epochs = [ep['epoch'] for ep in ldm_data['epochs']]
        ldm_losses = [ep['avg_loss'] for ep in ldm_data['epochs']]
        axes[1].plot(ldm_epochs, ldm_losses, 'r-', linewidth=2, label='LDM')
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Average Loss')
    axes[0].set_title('Autoencoder Training Progress')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Average Loss')
    axes[1].set_title('LDM Training Progress')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot training loss curves')
    parser.add_argument('--task_name', default='sar_colorization', 
                       help='Task name (directory containing losses)')
    parser.add_argument('--type', choices=['autoencoder', 'ldm', 'both'], default='both',
                       help='Type of losses to plot')
    parser.add_argument('--save', action='store_true',
                       help='Save plots to files')
    parser.add_argument('--output_dir', default='plots',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Create output directory if saving
    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
    
    autoencoder_data = None
    ldm_data = None
    
    # Load data based on type
    if args.type in ['autoencoder', 'both']:
        autoencoder_data = load_loss_data(args.task_name, 'autoencoder')
        if autoencoder_data:
            save_path = os.path.join(args.output_dir, 'autoencoder_losses.png') if args.save else None
            plot_autoencoder_losses(autoencoder_data, save_path)
        else:
            print("No autoencoder loss data found")
    
    if args.type in ['ldm', 'both']:
        ldm_data = load_loss_data(args.task_name, 'ldm')
        if ldm_data:
            save_path = os.path.join(args.output_dir, 'ldm_losses.png') if args.save else None
            plot_ldm_losses(ldm_data, save_path)
        else:
            print("No LDM loss data found")
    
    # Plot comparison if both are available
    if args.type == 'both' and autoencoder_data and ldm_data:
        save_path = os.path.join(args.output_dir, 'training_comparison.png') if args.save else None
        plot_comparison(autoencoder_data, ldm_data, save_path)


def create_simple_plot_example():
    """
    Example function showing how to create simple plots from CSV data
    """
    print("\n" + "="*50)
    print("EXAMPLE: Simple plotting from CSV files")
    print("="*50)
    
    print("""
# Example: Load and plot from CSV files directly
import pandas as pd
import matplotlib.pyplot as plt

# Load autoencoder losses
ae_df = pd.read_csv('sar_colorization/losses/autoencoder_loss_history.csv')

# Simple epoch-level plot
epoch_avg = ae_df.groupby('epoch')['total_loss'].mean()
plt.figure(figsize=(10, 6))
plt.plot(epoch_avg.index, epoch_avg.values, 'b-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Average Total Loss')
plt.title('Autoencoder Training Progress')
plt.grid(True, alpha=0.3)
plt.show()

# Load LDM losses
ldm_df = pd.read_csv('sar_colorization/losses/ldm_loss_history.csv')

# Simple epoch-level plot
epoch_avg = ldm_df.groupby('epoch')['loss'].mean()
plt.figure(figsize=(10, 6))
plt.plot(epoch_avg.index, epoch_avg.values, 'r-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('LDM Training Progress')
plt.grid(True, alpha=0.3)
plt.show()
""")


if __name__ == '__main__':
    main()
    create_simple_plot_example()
