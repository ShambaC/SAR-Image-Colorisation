# SAR to Optical Image Translation

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
- Python 3.9+
- CUDA-capable GPU (recommended)
- At least 16GB RAM
- 50GB+ free disk space

### Dependencies Installation
```bash
# Install required packages
pip install -r requirements.txt

# Verify installation
python test_setup.py
```

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

### Train/Validation/Test Splits

The dataset automatically handles proper train/validation/test splits with the following features:

#### Automatic Splitting
- **Default ratios**: 70% train, 15% validation, 15% test
- **Reproducible**: Uses fixed random seed for consistent splits across runs
- **No data leakage**: Ensures complete separation between splits
- **Stratified by source**: Maintains region/season distribution across splits

#### Split Configuration
Configure splits in `config/sar_config.yaml`:
```yaml
dataset_params:
  train_split: 0.7    # 70% for training
  val_split: 0.15     # 15% for validation  
  test_split: 0.15    # 15% for testing
```

#### Split Methods
The system supports three split methods:

1. **Auto-split from all data** (recommended):
   - Place all CSV files in Dataset folder
   - System automatically creates reproducible splits

2. **Pre-split files**:
   - Create `train_metadata.csv`, `val_metadata.csv`, `test_metadata.csv`
   - System will use these directly

3. **Single file splitting**:
   - Use single CSV file, specify with `metadata_file` parameter
   - System splits it according to ratios

### Data Validation

#### Validate Dataset Structure
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

#### Validate Train/Test Splits
Test your data splitting setup:
```bash
python validate_splits.py
```

This script will:
- Verify split proportions are correct
- Check for data leakage between splits
- Test reproducibility of splits
- Analyze data distribution across splits

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
  ldm_batch_size: 64          # Batch size for diffusion training
  autoencoder_batch_size: 4  # Batch size for autoencoder training
  ldm_epochs: 60           # Number of diffusion training epochs
  autoencoder_epochs: 25    # Number of autoencoder training epochs
  ldm_lr: 1e-4         # Learning rate for diffusion model
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

## Model evaluation

**Metrics Computed**:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity) - if available
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)

**Usage**:
```bash
# Full evaluation with default settings
python evaluate_model.py --config config/sar_config.yaml

# Evaluate specific number of samples with custom guidance scale
python evaluate_model.py --config config/sar_config.yaml --num_samples 50 --guidance_scale 7.5

# Skip saving sample images to save disk space
python evaluate_model.py --config config/sar_config.yaml --no_save_samples
```

**Output**: Results saved in `sar_colorization/evaluation_results/`

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

## Model Validation and Testing

### During Training
The training process automatically uses:
- **Training set**: For model parameter updates
- **Validation set**: For monitoring overfitting (if implemented)
- **Test set**: Reserved for final evaluation

### Test Set Evaluation
After training, evaluate your model on the test set:

```python
# Example test set evaluation
from dataset.sar_dataset import SARDataset
from torch.utils.data import DataLoader

# Load test dataset
test_dataset = SARDataset(
    split='test',
    im_path='../Dataset',
    im_size=256,
    im_channels=3,
    condition_config={'condition_types': ['text', 'image']},
    train_split=0.7,
    val_split=0.15,
    test_split=0.15,
    random_seed=42
)

test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Run inference on test set
for batch in test_loader:
    optical_images, conditions = batch
    # Your evaluation code here
    pass
```

### Split Integrity
The system ensures:
- ✅ **No data leakage**: Test images never seen during training
- ✅ **Reproducible splits**: Same random seed = same splits
- ✅ **Balanced distribution**: Regions/seasons distributed across splits
- ✅ **Proper proportions**: Configurable train/val/test ratios

## Support and Resources

- Check console outputs for detailed error messages
- Monitor GPU memory usage with `nvidia-smi`
- Use TensorBoard for training visualization (optional integration)
- Refer to original Stable Diffusion papers for theoretical background

For additional help, ensure your environment matches the requirements and that the dataset structure is exactly as specified.
