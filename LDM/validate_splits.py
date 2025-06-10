"""
Validation script to test train/val/test splits and ensure proper data separation.
This script helps verify that the splitting functionality works correctly.
"""
import yaml
import torch
import os
import numpy as np
from dataset.sar_dataset import SARDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


def test_splits_integrity(config):
    """
    Test that train/val/test splits are properly separated and consistent
    """
    print("=== Testing Data Split Integrity ===")
    
    dataset_config = config['dataset_params']
    condition_config = config['ldm_params']['condition_config']
    
    # Test parameters
    test_params = {
        'im_path': dataset_config['im_path'],
        'im_size': dataset_config['im_size'], 
        'im_channels': dataset_config['im_channels'],
        'use_latents': False,
        'condition_config': condition_config,
        'train_split': dataset_config.get('train_split', 0.7),
        'val_split': dataset_config.get('val_split', 0.15),
        'test_split': dataset_config.get('test_split', 0.15),
        'random_seed': 42  # Fixed seed for consistency
    }
    
    # Create datasets for each split
    splits = ['train', 'val', 'test']
    datasets = {}
    
    try:
        for split in splits:
            print(f"\nLoading {split} dataset...")
            datasets[split] = SARDataset(split=split, **test_params)
            print(f"  - {split} dataset size: {len(datasets[split])}")
        
        # Check split proportions
        total_samples = sum(len(datasets[split]) for split in splits)
        print(f"\nDataset statistics:")
        print(f"  - Total samples: {total_samples}")
        
        for split in splits:
            proportion = len(datasets[split]) / total_samples
            expected = test_params[f'{split}_split']
            print(f"  - {split}: {len(datasets[split])} samples ({proportion:.3f}, expected: {expected:.3f})")
        
        # Test reproducibility - create datasets again with same seed
        print("\nTesting reproducibility...")
        datasets_2 = {}
        for split in splits:
            datasets_2[split] = SARDataset(split=split, **test_params)
        
        # Compare first few samples
        all_consistent = True
        for split in splits:
            if len(datasets[split]) > 0 and len(datasets_2[split]) > 0:
                # Compare SAR image paths (should be identical)
                sample_idx = min(2, len(datasets[split]) - 1)
                path1 = datasets[split].s1_images[sample_idx]
                path2 = datasets_2[split].s1_images[sample_idx]
                if path1 != path2:
                    print(f"  ✗ {split} split not reproducible!")
                    all_consistent = False
                    break
        
        if all_consistent:
            print("  ✓ Splits are reproducible with same random seed")
        
        # Test data separation (no overlap between splits)
        print("\nTesting data separation...")
        train_images = set(datasets['train'].s1_images)
        val_images = set(datasets['val'].s1_images)
        test_images = set(datasets['test'].s1_images)
        
        overlaps = []
        if train_images & val_images:
            overlaps.append("train-val")
        if train_images & test_images:
            overlaps.append("train-test")
        if val_images & test_images:
            overlaps.append("val-test")
        
        if overlaps:
            print(f"  ✗ Found data leakage between: {', '.join(overlaps)}")
            return False
        else:
            print("  ✓ No data leakage detected - splits are properly separated")
        
        # Test sample loading
        print("\nTesting sample loading...")
        for split in splits:
            if len(datasets[split]) > 0:
                sample = datasets[split][0]
                if isinstance(sample, tuple):
                    image, cond_input = sample
                    print(f"  - {split} sample: image shape {image.shape}, conditions {list(cond_input.keys())}")
                else:
                    print(f"  - {split} sample: image shape {sample.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during split testing: {e}")
        return False


def evaluate_on_test_set(config, model_path=None):
    """
    Example of how to evaluate trained model on test set
    """
    print("\n=== Test Set Evaluation Example ===")
    
    if not model_path or not os.path.exists(model_path):
        print("No trained model found. This is just a demonstration of test set loading.")
        print("Train your model first using: python train_model.py")
        
        # Just demonstrate test set loading
        dataset_config = config['dataset_params']
        condition_config = config['ldm_params']['condition_config']
        
        test_dataset = SARDataset(
            split='test',
            im_path=dataset_config['im_path'],
            im_size=dataset_config['im_size'],
            im_channels=dataset_config['im_channels'],
            use_latents=False,
            condition_config=condition_config,
            train_split=dataset_config.get('train_split', 0.7),
            val_split=dataset_config.get('val_split', 0.15),
            test_split=dataset_config.get('test_split', 0.15),
            random_seed=42        )
        
        print(f"Test dataset loaded: {len(test_dataset)} samples")
        
        if len(test_dataset) > 0:
            # Show example of iterating through test set
            test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
            
            print("Example test batch:")
            for batch_idx, batch in enumerate(test_loader):
                if isinstance(batch, tuple):
                    images, cond_inputs = batch
                    print(f"  - Batch {batch_idx}: {images.shape[0]} samples")
                    print(f"  - Sample text prompts: {cond_inputs['text'][:2]}")
                elif isinstance(batch, torch.Tensor):
                    print(f"  - Batch {batch_idx}: {batch.shape[0]} samples")
                else:
                    # Handle case where batch might be a list or other structure
                    print(f"  - Batch {batch_idx}: batch type = {type(batch)}")
                    if hasattr(batch, '__len__'):
                        print(f"  - Batch size: {len(batch)}")
                
                if batch_idx >= 2:  # Just show first few batches
                    break
        
        return
    
    # If model exists, could add actual evaluation code here
    print(f"Model evaluation with {model_path} would go here...")


def visualize_split_distribution(config):
    """
    Create visualizations of the data splits
    """
    print("\n=== Split Distribution Analysis ===")
    
    try:
        dataset_config = config['dataset_params']
        condition_config = config['ldm_params']['condition_config']
        
        # Load all splits
        splits_data = {}
        for split in ['train', 'val', 'test']:
            dataset = SARDataset(
                split=split,
                im_path=dataset_config['im_path'],
                im_size=dataset_config['im_size'],
                im_channels=dataset_config['im_channels'],
                use_latents=False,
                condition_config=condition_config,
                train_split=dataset_config.get('train_split', 0.7),
                val_split=dataset_config.get('val_split', 0.15),
                test_split=dataset_config.get('test_split', 0.15),
                random_seed=42
            )
            
            splits_data[split] = {
                'size': len(dataset),
                'text_prompts': dataset.text_prompts if len(dataset) > 0 else []
            }
        
        # Print summary
        total = sum(data['size'] for data in splits_data.values())
        print(f"Total dataset size: {total}")
        
        for split, data in splits_data.items():
            percentage = (data['size'] / total * 100) if total > 0 else 0
            print(f"{split.capitalize()}: {data['size']} samples ({percentage:.1f}%)")
        
        # Analyze text prompt distribution if available
        print("\nText prompt analysis:")
        for split, data in splits_data.items():
            if data['text_prompts']:
                regions = []
                seasons = []
                for prompt in data['text_prompts']:
                    # Extract region and season from prompt
                    parts = prompt.split(', ')
                    for part in parts:
                        if part.startswith('Region: '):
                            regions.append(part.replace('Region: ', ''))
                        elif part.startswith('Season: '):
                            seasons.append(part.replace('Season: ', ''))
                
                unique_regions = set(regions)
                unique_seasons = set(seasons)
                print(f"  {split}: {len(unique_regions)} regions, {len(unique_seasons)} seasons")
                
    except Exception as e:
        print(f"Error in distribution analysis: {e}")


def main():
    # Load configuration
    config_path = 'config/sar_config.yaml'
    
    print("Loading configuration...")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    print(f"Dataset path: {config['dataset_params']['im_path']}")
    
    # Check if dataset exists
    dataset_path = config['dataset_params']['im_path']
    if not os.path.exists(dataset_path):
        print(f"Dataset path '{dataset_path}' does not exist.")
        print("Please update the dataset path in config/sar_config.yaml or prepare your dataset.")
        return
    
    # Run tests
    success = test_splits_integrity(config)
    
    if success:
        visualize_split_distribution(config)
        evaluate_on_test_set(config)
        print("\n" + "="*50)
        print("✓ SPLIT VALIDATION COMPLETE")
        print("="*50)
        print("\nYour data splitting is working correctly!")
        print("You can now train your model with proper train/val/test splits.")
    else:
        print("\n" + "="*50)
        print("✗ SPLIT VALIDATION FAILED")
        print("="*50)
        print("Please check your dataset and fix any issues before training.")


if __name__ == '__main__':
    main()
