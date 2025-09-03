import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot(config_path):
    # Load config to get log file paths
    with open(config_path, 'r') as f:
        import yaml
        config = yaml.safe_load(f)

    # Plot VAE Losses
    try:
        vae_df = pd.read_csv(config['vae_log_path'])
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(vae_df['epoch'], vae_df['avg_recon_loss'], label='Reconstruction Loss')
        plt.plot(vae_df['epoch'], vae_df['avg_kl_div'], label='KL Divergence')
        plt.title('VAE Loss Components')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(vae_df['epoch'], vae_df['avg_loss'], label='Total VAE Loss', color='red')
        plt.title('Total VAE Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.suptitle('VAE Training Performance')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    except FileNotFoundError:
        print(f"VAE log file not found at {config['vae_log_path']}")

    # Plot Diffusion Losses
    try:
        diffusion_df = pd.read_csv(config['diffusion_log_path'])
        plt.figure(figsize=(8, 5))
        plt.plot(diffusion_df['epoch'], diffusion_df['avg_loss'], label='Diffusion MSE Loss', color='purple')
        plt.title('Diffusion Model Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
    except FileNotFoundError:
        print(f"Diffusion log file not found at {config['diffusion_log_path']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot training losses from CSV logs.")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to the config file.")
    args = parser.parse_args()
    plot(args.config)