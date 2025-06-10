# SAR Model Evaluation Scripts

This directory contains comprehensive evaluation scripts for the SAR to optical image translation model. The scripts provide different levels of evaluation detail and comparison capabilities.

## Available Scripts

### 1. `quick_evaluate.py` - Fast VQVAE Reconstruction Evaluation
**Purpose**: Quickly evaluate VQVAE reconstruction quality without running the full diffusion process.

**Metrics Computed**:
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error) 
- PSNR (Peak Signal-to-Noise Ratio)
- Quantization losses (codebook and commitment)

**Usage**:
```bash
# Evaluate all test samples
python quick_evaluate.py --config config/sar_config.yaml

# Evaluate specific number of samples
python quick_evaluate.py --config config/sar_config.yaml --num_samples 100
```

**Output**: Results saved in `sar_colorization/quick_evaluation_results/`

---

### 2. `evaluate_model.py` - Complete Pipeline Evaluation
**Purpose**: Full evaluation including diffusion sampling to generate optical images from SAR inputs.

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

---

### 3. `compare_models.py` - Model Comparison
**Purpose**: Compare multiple model checkpoints or different configurations on the same test dataset.

**Usage**:
```bash
# Compare different checkpoints
python compare_models.py --config config/sar_config.yaml \
    --checkpoints sar_colorization/vqvae_autoencoder_ckpt.pth \
                  sar_colorization/backup/vqvae_epoch_15.pth \
    --names "Final_Model" "Epoch_15" \
    --num_samples 200

# Compare with automatic naming
python compare_models.py --config config/sar_config.yaml \
    --checkpoints checkpoint1.pth checkpoint2.pth checkpoint3.pth
```

**Output**: Results saved in `model_comparison_results/`

---

### 4. `run_evaluation.py` - Batch Evaluation Runner
**Purpose**: Convenient script to run multiple evaluation types in sequence.

**Usage**:
```bash
# Run all evaluations
python run_evaluation.py --action all --config config/sar_config.yaml

# Run only quick evaluation
python run_evaluation.py --action quick --config config/sar_config.yaml --num_samples 100

# Run only full evaluation
python run_evaluation.py --action full --config config/sar_config.yaml

# Compare models
python run_evaluation.py --action compare --config config/sar_config.yaml \
    --checkpoints model1.pth model2.pth --names "Model_A" "Model_B"
```

## Output Structure

### Quick Evaluation Results
```
sar_colorization/quick_evaluation_results/
├── quick_evaluation_report.json          # Detailed JSON report
├── reconstruction_metrics.png            # Metrics distribution plots
├── quantization_losses.png              # VQ losses over batches
└── vqvae_reconstruction_samples.png     # Sample reconstructions
```

### Full Evaluation Results
```
sar_colorization/evaluation_results/
├── evaluation_report.json               # Comprehensive JSON report
├── metrics_distribution.png            # Histogram of all metrics
├── metrics_per_sample.png              # Time series of metrics
└── sample_images/                       # Sample generated images
    ├── sample_000_comparison.png        # SAR | Generated | Ground Truth
    ├── sample_001_comparison.png
    └── ...
```

### Model Comparison Results
```
model_comparison_results/
├── comparison_report.json               # JSON comparison report
├── model_comparison_summary.csv         # Summary statistics table
├── model_comparison_detailed.csv        # Per-sample detailed results
├── metrics_comparison_bars.png          # Bar chart comparison
├── metrics_comparison_boxes.png         # Box plot distributions
└── metrics_comparison_error_bars.png    # Error bar comparison
```

## Understanding the Metrics

### Image Quality Metrics
- **PSNR**: Peak Signal-to-Noise Ratio (higher is better, >20 dB is good)
- **SSIM**: Structural Similarity Index (0-1, higher is better, >0.8 is good)
- **MSE**: Mean Squared Error (lower is better, <0.01 is good)
- **MAE**: Mean Absolute Error (lower is better, <0.05 is good)
- **LPIPS**: Learned Perceptual Similarity (lower is better, <0.3 is good)

### VQVAE Specific Metrics
- **Codebook Loss**: Measures how well the codebook represents the data
- **Commitment Loss**: Encourages encoder outputs to commit to codebook entries

## Requirements

Make sure you have the following packages installed:
```bash
pip install matplotlib seaborn scikit-image scikit-learn pandas
pip install lpips  # Optional, for LPIPS metric
```

## Tips for Better Evaluation

1. **Sample Size**: Start with a smaller number of samples (e.g., 100) for quick testing, then run full evaluation on the complete test set.

2. **Memory Management**: If you run out of GPU memory, reduce the batch size in the DataLoader or evaluate fewer samples at a time.

3. **Metric Selection**: 
   - Use **quick evaluation** for rapid iteration during development
   - Use **full evaluation** for final model assessment
   - Use **model comparison** when choosing between different approaches

4. **Baseline Comparison**: Always compare against a simple baseline (e.g., nearest neighbor upsampling) to ensure your model is learning meaningful representations.

## Troubleshooting

### Common Issues

1. **"No module named 'lpips'"**: LPIPS is optional. Install with `pip install lpips` or the evaluation will skip this metric.

2. **CUDA out of memory**: Reduce batch size in the DataLoader or use `--num_samples` to evaluate fewer samples.

3. **"Test dataset empty"**: Check that your dataset path and metadata files are correctly configured in `sar_config.yaml`.

4. **"Checkpoint not found"**: Ensure model training has completed and checkpoint files exist in the specified paths.

### Performance Expectations

Typical performance ranges for SAR to optical translation:
- **PSNR**: 15-25 dB (higher is better)
- **SSIM**: 0.6-0.9 (higher is better)  
- **MSE**: 0.001-0.1 (lower is better)
- **MAE**: 0.01-0.2 (lower is better)

These ranges depend heavily on your dataset quality and model architecture.
