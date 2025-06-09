# SAR Image Colorization Training and Inference Guide

This guide provides comprehensive instructions for training and using the SAR (Synthetic Aperture Radar) to optical image colorization model.

## Table of Contents

1. [Dataset Structure](#dataset-structure)
2. [Environment Setup](#environment-setup)
3. [Training CLIP Model](#training-clip-model)
4. [Training Main Model](#training-main-model)
5. [Inference](#inference)
6. [Configuration Guide](#configuration-guide)
7. [Hyperparameter Tuning](#hyperparameter-tuning)
8. [Troubleshooting](#troubleshooting)

## Dataset Structure

The expected dataset structure is:

```
Dataset/
├── r_000/
│   ├── s1_000/
│   │   ├── img_p0.png
│   │   ├── img_p1.png
│   │   └── ...
│   └── s2_000/
│       ├── img_p0.png
│       ├── img_p1.png
│       └── ...
├── r_001/
│   ├── s1_001/
│   └── s2_001/
├── data_r_000.csv
├── data_r_001.csv
└── ...
```

### CSV File Format

Each `data_r_XXX.csv` file contains metadata with the following columns:
- `s1_fileName`: Path to SAR image (relative to Dataset folder)
- `s2_fileName`: Path to optical image (relative to Dataset folder)
- `coordinates`: Geographic coordinates
- `country`: Country name
- `date-time`: Timestamp
- `scale`: Image scale
- `region`: Region type (tropical, temperate, arctic)
- `season`: Season (winter, spring, summer, autumn/fall)
- `operational-mode`: SAR operational mode
- `polarisation`: SAR polarization
- `bands`: Spectral bands information

## Environment Setup

### 1. Install Dependencies

```powershell
# Create and activate virtual environment (if not already done)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 2. Verify Dataset

```powershell
# Test dataset loading
python dataset.py
```

This will show dataset statistics and verify that images can be loaded correctly.

## Training CLIP Model

The CLIP model learns to encode text prompts into embeddings that guide the image generation process.

### 1. Basic CLIP Training

```powershell
python train_clip.py --dataset_path ../Dataset --config_file configs/clip_config.json
```

### 2. CLIP Training with Custom Parameters

```powershell
python train_clip.py `
    --dataset_path ../Dataset `
    --batch_size 32 `
    --learning_rate 1e-4 `
    --epochs 50 `
    --save_dir ./checkpoints/clip `
    --log_dir ./logs/clip `
    --vocab_size 1000 `
    --embedding_dim 768 `
    --max_length 77 `
    --num_heads 12 `
    --num_layers 12
```

### 3. Resume CLIP Training

```powershell
python train_clip.py `
    --dataset_path ../Dataset `
    --resume_from ./checkpoints/clip/clip_epoch_10.pt
```

### 4. CLIP Training Configuration

Create `configs/clip_config.json`:

```json
{
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
        "save_every": 5
    },
    "data": {
        "train_split": 0.8,
        "num_workers": 4,
        "pin_memory": true
    }
}
```

## Training Main Model

The main model combines VAE encoding, diffusion process, and CLIP guidance for image-to-image translation.

### 1. Basic Training

```powershell
python train_model.py `
    --dataset_path ../Dataset `
    --clip_checkpoint ./checkpoints/clip/best_clip.pt `
    --config_file configs/model_config.json
```

### 2. Training with Custom Parameters

```powershell
python train_model.py `
    --dataset_path ../Dataset `
    --clip_checkpoint ./checkpoints/clip/best_clip.pt `
    --batch_size 4 `
    --learning_rate 1e-4 `
    --epochs 100 `
    --save_dir ./checkpoints/model `
    --log_dir ./logs/model `
    --mixed_precision `
    --gradient_accumulation_steps 4
```

### 3. Resume Training

```powershell
python train_model.py `
    --dataset_path ../Dataset `
    --resume_from ./checkpoints/model/model_epoch_20.pt
```

### 4. Main Model Configuration

Create `configs/model_config.json`:

```json
{
    "clip_config": {
        "n_vocab": 1000,
        "n_embd": 768,
        "max_length": 77,
        "n_head": 12,
        "n_layers": 12
    },
    "diffusion_config": {
        "in_channels": 8,
        "out_channels": 4,
        "model_channels": 320,
        "attention_resolutions": [4, 2, 1],
        "num_res_blocks": 2,
        "channel_mult": [1, 2, 4, 4],
        "num_heads": 8,
        "use_spatial_transformer": true,
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
        "mixed_precision": true,
        "gradient_accumulation_steps": 4,
        "max_grad_norm": 1.0
    },
    "data": {
        "train_split": 0.8,
        "num_workers": 4,
        "pin_memory": true
    }
}
```

## Inference

### 1. Single Image Inference

```powershell
python infer_model.py `
    --mode single `
    --model_checkpoint ./checkpoints/model/best_model.pt `
    --clip_checkpoint ./checkpoints/clip/best_clip.pt `
    --input_image path/to/sar_image.png `
    --region tropical `
    --season summer `
    --output_path generated_image.png `
    --guidance_scale 7.5 `
    --num_inference_steps 50
```

### 2. Batch Inference

```powershell
python infer_model.py `
    --mode batch `
    --model_checkpoint ./checkpoints/model/best_model.pt `
    --clip_checkpoint ./checkpoints/clip/best_clip.pt `
    --input_dir ./test_images `
    --regions tropical temperate arctic `
    --seasons winter spring summer autumn `
    --output_dir ./generated_images `
    --guidance_scale 7.5 `
    --num_inference_steps 50
```

### 3. Dataset Inference

```powershell
python infer_model.py `
    --mode dataset `
    --model_checkpoint ./checkpoints/model/best_model.pt `
    --clip_checkpoint ./checkpoints/clip/best_clip.pt `
    --dataset_path ../Dataset `
    --output_dir ./dataset_results `
    --num_samples 20 `
    --guidance_scale 7.5 `
    --num_inference_steps 50
```

### 4. Inference Parameters

- `--strength`: Controls how much the input image is modified (0.0-1.0, default: 0.8)
- `--guidance_scale`: Classifier-free guidance strength (1.0-20.0, default: 7.5)
- `--num_inference_steps`: Number of denoising steps (10-100, default: 50)
- `--seed`: Random seed for reproducible results

## Configuration Guide

### CLIP Model Parameters

- **vocab_size**: Size of vocabulary (default: 1000)
  - Increase for more diverse text understanding
  - Decrease to reduce model size

- **embedding_dim**: Embedding dimension (default: 768)
  - Standard sizes: 512, 768, 1024
  - Larger dimensions for better representation

- **max_length**: Maximum sequence length (default: 77)
  - Should accommodate longest expected prompts

- **num_heads**: Number of attention heads (default: 12)
  - Must divide embedding_dim evenly
  - More heads for better attention patterns

- **num_layers**: Number of transformer layers (default: 12)
  - More layers for better text understanding
  - Increases training time and memory usage

### Diffusion Model Parameters

- **model_channels**: Base channel dimension (default: 320)
  - Affects model capacity and quality
  - Common values: 256, 320, 512

- **channel_mult**: Channel multipliers per resolution level
  - Controls how channels scale with resolution
  - Example: [1, 2, 4, 4] doubles channels each level

- **num_res_blocks**: Residual blocks per level (default: 2)
  - More blocks for better representation
  - Increases computation time

- **attention_resolutions**: Which levels use attention
  - Lower values = higher resolution attention
  - Example: [4, 2, 1] uses attention at 1/4, 1/2, full res

### Training Parameters

- **batch_size**: Training batch size
  - Larger batches for stability (if memory allows)
  - Typical range: 4-32 for diffusion models

- **learning_rate**: Optimizer learning rate
  - CLIP: 1e-4 to 5e-4
  - Diffusion: 1e-5 to 1e-4

- **gradient_accumulation_steps**: Accumulate gradients
  - Use when batch size is limited by memory
  - Effective batch size = batch_size × accumulation_steps

## Hyperparameter Tuning

### 1. CLIP Model Tuning

**Learning Rate Schedule:**
```json
{
    "learning_rate": 1e-4,
    "lr_scheduler": "cosine",
    "warmup_steps": 1000,
    "min_lr": 1e-6
}
```

**Regularization:**
```json
{
    "weight_decay": 0.01,
    "dropout": 0.1,
    "grad_clip_norm": 1.0
}
```

### 2. Diffusion Model Tuning

**Training Schedule:**
```json
{
    "learning_rate": 1e-4,
    "beta1": 0.9,
    "beta2": 0.999,
    "eps": 1e-8,
    "weight_decay": 0.01
}
```

**Loss Weighting:**
```json
{
    "mse_loss_weight": 1.0,
    "perceptual_loss_weight": 0.1,
    "adversarial_loss_weight": 0.01
}
```

### 3. Data Augmentation

```json
{
    "augmentation": {
        "horizontal_flip": 0.5,
        "rotation": 10,
        "color_jitter": {
            "brightness": 0.1,
            "contrast": 0.1,
            "saturation": 0.1,
            "hue": 0.05
        },
        "gaussian_noise": 0.01
    }
}
```

## Performance Optimization

### 1. Memory Optimization

- Use mixed precision training (`--mixed_precision`)
- Reduce batch size and increase gradient accumulation
- Use gradient checkpointing for large models
- Clear cache regularly: `torch.cuda.empty_cache()`

### 2. Speed Optimization

- Increase `num_workers` for data loading
- Use `pin_memory=True` for GPU training
- Compile models with `torch.compile()` (PyTorch 2.0+)
- Use efficient attention implementations

### 3. Quality Optimization

- Train CLIP model thoroughly before main model
- Use higher resolution during training if possible
- Experiment with different loss combinations
- Use validation set for early stopping

## Monitoring Training

### 1. TensorBoard Logs

```powershell
# Start TensorBoard
tensorboard --logdir ./logs
```

Key metrics to monitor:
- Training loss trends
- Validation loss
- Generated image samples
- Learning rate schedules

### 2. Checkpointing

Models are automatically saved at specified intervals:
- `best_model.pt`: Best validation performance
- `latest_model.pt`: Most recent checkpoint
- `model_epoch_X.pt`: Epoch-specific checkpoints

## Troubleshooting

### Common Issues

**1. Out of Memory Errors**
```powershell
# Reduce batch size
--batch_size 2

# Increase gradient accumulation
--gradient_accumulation_steps 8

# Use mixed precision
--mixed_precision
```

**2. Training Instability**
```powershell
# Reduce learning rate
--learning_rate 5e-5

# Add gradient clipping
--max_grad_norm 0.5

# Increase warmup steps
--warmup_steps 2000
```

**3. Poor Generation Quality**
- Ensure CLIP model is well-trained
- Check that text prompts match training data
- Increase guidance scale for stronger text conditioning
- Verify dataset quality and preprocessing

**4. Slow Training**
```powershell
# Increase number of workers
--num_workers 8

# Use pin memory
--pin_memory

# Reduce validation frequency
--validate_every 10
```

### Debug Commands

**Check dataset statistics:**
```powershell
python dataset.py
```

**Test model loading:**
```powershell
python -c "from train_model import *; print('Models loaded successfully')"
```

**Validate configuration:**
```powershell
python train_model.py --dataset_path ../Dataset --config_file configs/model_config.json --dry_run
```

## Advanced Usage

### 1. Custom Loss Functions

Modify `train_model.py` to implement custom losses:
- Perceptual loss using pre-trained VGG
- LPIPS loss for better perceptual quality
- Adversarial loss with discriminator

### 2. Multi-GPU Training

```powershell
# DataParallel (single machine, multiple GPUs)
python train_model.py --multi_gpu

# DistributedDataParallel (multiple machines)
python -m torch.distributed.launch --nproc_per_node=4 train_model.py --distributed
```

### 3. Custom Text Prompts

Create custom prompt templates in `dataset.py`:
```python
def generate_custom_prompt(metadata):
    return f"Transform SAR image to optical in {metadata['season']} at {metadata['region']} region with {metadata['weather']} conditions"
```

### 4. Model Architecture Modifications

- Modify attention mechanisms in `attention.py`
- Customize U-Net architecture in `diffusion.py`
- Implement new conditioning methods in `clip.py`

## Best Practices

1. **Training Order**: Always train CLIP model first, then the main diffusion model
2. **Data Quality**: Ensure SAR and optical images are properly aligned
3. **Validation**: Use a held-out validation set for model selection
4. **Experimentation**: Keep detailed logs of hyperparameter experiments
5. **Reproducibility**: Set random seeds for consistent results
6. **Backup**: Regularly backup model checkpoints and training logs

## Example Training Scripts

### Full Training Pipeline

```powershell
# 1. Train CLIP model
python train_clip.py `
    --dataset_path ../Dataset `
    --batch_size 32 `
    --epochs 50 `
    --learning_rate 1e-4 `
    --save_dir ./checkpoints/clip

# 2. Train main model
python train_model.py `
    --dataset_path ../Dataset `
    --clip_checkpoint ./checkpoints/clip/best_clip.pt `
    --batch_size 4 `
    --epochs 100 `
    --learning_rate 1e-4 `
    --mixed_precision `
    --save_dir ./checkpoints/model

# 3. Run inference
python infer_model.py `
    --mode dataset `
    --model_checkpoint ./checkpoints/model/best_model.pt `
    --clip_checkpoint ./checkpoints/clip/best_clip.pt `
    --dataset_path ../Dataset `
    --output_dir ./results `
    --num_samples 50
```

This guide provides a comprehensive overview of training and using the SAR image colorization system. For specific questions or issues, refer to the code documentation or create detailed error reports.
