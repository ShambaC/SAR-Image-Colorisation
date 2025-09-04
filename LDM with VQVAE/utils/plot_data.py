import pandas as pd
import matplotlib.pyplot as plt

task_name = "sar_image_colorisation_ldm"

# Load VQVAE stats
vqvae_stats = pd.read_csv(f'{task_name}/vqvae_training_stats.csv')
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(vqvae_stats['epoch'], vqvae_stats['recon_loss'])
plt.title('Reconstruction Loss')

# Load LDM stats
ldm_stats = pd.read_csv(f'{task_name}/ldm_training_stats.csv')
plt.subplot(2, 2, 2)
plt.plot(ldm_stats['epoch'], ldm_stats['loss'])
plt.title('Diffusion Loss')