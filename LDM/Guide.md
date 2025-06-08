# SAR to Optical Image Translation Guide

This guide provides comprehensive instructions for training and using the SAR to Optical image translation model using text-guided latent diffusion.

## Table of Contents
1. [Setup and Installation](#setup-and-installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Configuration](#configuration)
4. [Training](#training)
5. [Inference](#inference)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Troubleshooting](#troubleshooting)

## Setup and Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- At least 16GB RAM
- 50GB+ free disk space

### Dependencies Installation
```bash
# Install required packages
pip install pandas torch torchvision transformers pillow tqdm pyyaml numpy matplotlib

# Verify installation
python test_setup.py
```

### Installation Steps

1. **Create and activate conda environment:**
```bash
conda create -n sar_colorization python=3.8
conda activate sar_colorization
```

2. **Install required packages:**
```bash
pip install -r requirements.txt
```

3. **Download LPIPS weights:**
   - Open this link in browser: https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/weights/v0.1/vgg.pth
   - Download the raw file and place it at `models/weights/v0.1/vgg.pth`

## Dataset Preparation

### Dataset Structure
Ensure your dataset follows this structure:
```
Dataset/
├── r_000/
│   ├── s1_000/
│   │   └── img_p0.png, img_p1.png, ...
│   └── s2_000/
│       └── img_p0.png, img_p1.png, ...
├── r_001/
│   ├── s1_001/
│   └── s2_001/
├── data_r_000.csv
├── data_r_001.csv
└── ...
```

### CSV File Format
Each `data_r_XXX.csv` should contain:
- `s1_fileName`: Path to SAR image relative to Dataset folder
- `s2_fileName`: Path to optical image relative to Dataset folder
- `region`: Geographic region description
- `season`: Seasonal information
- Other metadata columns (optional)

### Data Validation
Run this script to validate your dataset:
```python
import os
import pandas as pd
import glob

def validate_dataset(dataset_path):
    csv_files = glob.glob(os.path.join(dataset_path, 'data_r_*.csv'))
    total_pairs = 0
    missing_files = 0
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            s1_path = os.path.join(dataset_path, row['s1_fileName'])
            s2_path = os.path.join(dataset_path, row['s2_fileName'])
            
            if not os.path.exists(s1_path) or not os.path.exists(s2_path):
                missing_files += 1
                print(f"Missing: {s1_path} or {s2_path}")
            total_pairs += 1
    
    print(f"Total pairs: {total_pairs}")
    print(f"Missing files: {missing_files}")
    print(f"Valid pairs: {total_pairs - missing_files}")

# Run validation
validate_dataset("../Dataset")
```

## Configuration

### Main Configuration File: `config/sar_config.yaml`

#### Dataset Parameters
```yaml
dataset_params:
  im_path: '../Dataset'  # Path to your dataset
  im_channels: 3         # RGB channels
  im_size: 256          # Image resolution
  name: 'sar'           # Dataset identifier
```

#### Model Architecture
```yaml
autoencoder_params:
  z_channels: 4               # Latent space channels
  codebook_size: 8192        # VQ codebook size
  down_channels: [64, 128, 256, 256]  # Encoder channels
  # ... other parameters
```

#### Training Parameters
```yaml
train_params:
  ldm_batch_size: 8          # Batch size for diffusion training
  autoencoder_batch_size: 4  # Batch size for autoencoder training
  ldm_epochs: 100           # Number of diffusion training epochs
  autoencoder_epochs: 30    # Number of autoencoder training epochs
  ldm_lr: 0.000005         # Learning rate for diffusion model
  autoencoder_lr: 0.00001  # Learning rate for autoencoder
  cf_guidance_scale: 7.5   # Classifier-free guidance scale
  # ... other parameters
```

## Training

### Stage 1: Train Autoencoder (VQVAE)

Train the autoencoder on optical images:
```bash
python train_model.py --config config/sar_config.yaml --stage autoencoder
```

**Key points:**
- Trains only on Sentinel-2 (optical) images
- Creates a compressed latent representation
- Typically takes 2-4 hours on modern GPUs
- Saves checkpoints in `sar_colorization/` folder

### Stage 2: Train Latent Diffusion Model

Train the diffusion model with conditioning:
```bash
python train_model.py --config config/sar_config.yaml --stage ldm
```

**Key points:**
- Uses SAR images + text prompts as conditioning
- Generates optical images in latent space
- Takes 8-12 hours depending on dataset size
- Requires completed autoencoder training

### Combined Training
Train both stages sequentially:
```bash
python train_model.py --config config/sar_config.yaml --stage both
```

### Monitoring Training

1. **Check loss curves:** Training progress is printed to console
2. **Sample images:** Autoencoder samples saved in `sar_colorization/autoencoder_samples/`
3. **Checkpoints:** Model weights saved in `sar_colorization/`

## Inference

### Interactive Mode (Recommended for testing)
```bash
python infer_model.py --config config/sar_config.yaml --mode interactive
```

### Single Image
```bash
python infer_model.py \
  --config config/sar_config.yaml \
  --mode single \
  --sar_image path/to/sar_image.png \
  --region "urban" \
  --season "summer" \
  --output colorized_result.png
```

### Batch Processing
```bash
python infer_model.py \
  --config config/sar_config.yaml \
  --mode batch \
  --sar_folder path/to/sar_images/ \
  --output results_folder/ \
  --region "forest" \
  --season "autumn"
```

### Custom Text Prompts
```bash
python infer_model.py \
  --config config/sar_config.yaml \
  --mode single \
  --sar_image image.png \
  --text_prompt "Colorise image, Region: urban, Season: winter"
```

## Hyperparameter Tuning

### Memory Optimization

**For Limited GPU Memory (< 8GB):**
```yaml
train_params:
  ldm_batch_size: 4          # Reduce batch size
  autoencoder_batch_size: 2
  autoencoder_acc_steps: 8   # Increase gradient accumulation
```

**For High-End GPUs (> 16GB):**
```yaml
train_params:
  ldm_batch_size: 16
  autoencoder_batch_size: 8
  autoencoder_acc_steps: 2
```

### Training Speed vs Quality

**Fast Training (Lower Quality):**
```yaml
train_params:
  ldm_epochs: 50
  autoencoder_epochs: 15
  im_size: 128               # Smaller images
diffusion_params:
  num_timesteps: 500         # Fewer diffusion steps
```

**High Quality (Slower Training):**
```yaml
train_params:
  ldm_epochs: 200
  autoencoder_epochs: 50
  im_size: 512               # Higher resolution
diffusion_params:
  num_timesteps: 1000
```

### Conditioning Strength

**Stronger Text Conditioning:**
```yaml
ldm_params:
  condition_config:
    text_condition_config:
      cond_drop_prob: 0.05   # Lower dropout = stronger conditioning
```

**Stronger Image Conditioning:**
```yaml
ldm_params:
  condition_config:
    image_condition_config:
      cond_drop_prob: 0.02
```

### Guidance Scale Tuning

- **Low guidance (2-5):** More diverse, less controlled outputs
- **Medium guidance (5-10):** Balanced quality and diversity  
- **High guidance (10-15):** More controlled but potentially less diverse

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```
Solution: Reduce batch sizes in config file
train_params:
  ldm_batch_size: 4
  autoencoder_batch_size: 2
```

**2. Dataset Loading Errors**
```
Error: "Dataset path does not exist"
Solution: Check relative path '../Dataset' or use absolute path
```

**3. Missing Checkpoints**
```
Error: "VQVAE checkpoint not found"
Solution: Train autoencoder first before LDM:
python train_model.py --stage autoencoder
```

**4. Poor Image Quality**
```
Solutions:
- Increase training epochs
- Adjust guidance scale during inference
- Check dataset quality and size
- Tune learning rates
```

**5. Text Conditioning Not Working**
```
Solutions:
- Verify text prompts are diverse enough
- Reduce cond_drop_prob in config
- Check text embedding model (CLIP vs BERT)
```

### Performance Optimization

**1. Enable Mixed Precision Training:**
Add to training script:
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

**2. Increase DataLoader Workers:**
```python
DataLoader(dataset, num_workers=4, pin_memory=True)
```

**3. Use Latent Caching:**
```yaml
train_params:
  save_latents: True  # Cache encoded images for faster training
```

### Evaluation Metrics

**Visual Quality Assessment:**
1. Generate sample images on validation set
2. Compare with ground truth optical images
3. Check for artifacts and consistency

**Quantitative Metrics (Optional):**
```python
# LPIPS (Learned Perceptual Image Patch Similarity)
# SSIM (Structural Similarity Index)
# FID (Fréchet Inception Distance)
```

## Advanced Usage

### Custom Text Embeddings
Modify conditioning for domain-specific vocabulary:
```yaml
ldm_params:
  condition_config:
    text_condition_config:
      text_embed_model: 'bert'  # Try BERT instead of CLIP
      text_embed_dim: 768
```

### Multi-Scale Training
For higher resolution images:
```yaml
dataset_params:
  im_size: 512
autoencoder_params:
  down_channels: [64, 128, 256, 512, 512]  # Add more layers
```

### Resume Training
```python
# In train_model.py, add checkpoint loading:
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    print("Resumed training from checkpoint")
```

## Tips for Best Results

1. **Dataset Quality:** Ensure good alignment between SAR and optical images
2. **Text Diversity:** Use varied text prompts during training
3. **Gradual Training:** Start with smaller images, then increase resolution
4. **Regular Validation:** Generate samples throughout training to monitor progress
5. **Hyperparameter Search:** Experiment with different learning rates and batch sizes

## Support and Resources

- Check console outputs for detailed error messages
- Monitor GPU memory usage with `nvidia-smi`
- Use TensorBoard for training visualization (optional integration)
- Refer to original Stable Diffusion papers for theoretical background

For additional help, ensure your environment matches the requirements and that the dataset structure is exactly as specified.
