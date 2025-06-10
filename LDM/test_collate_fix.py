#!/usr/bin/env python3
"""
Test script to verify the collate function fix
"""

import sys
import os
import yaml
import torch
from torch.utils.data import DataLoader

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset.sar_dataset import SARDataset

def custom_collate_fn(batch):
    """Custom collate function to handle SAR dataset batching"""
    # Handle single sample case (batch_size=1)
    if len(batch) == 1:
        sample = batch[0]
        if isinstance(sample, tuple):
            # (target_image, cond_inputs) format
            target, cond_inputs = sample
            
            # Convert target to batch format (add batch dimension)
            target_batch = target.unsqueeze(0)
            
            # Handle conditioning inputs - don't add batch dimension since we're working with single samples
            cond_batch = {}
            for key, value in cond_inputs.items():
                if key == 'text':
                    # Keep text as string (not list)
                    cond_batch[key] = value
                elif key == 'image':
                    # Keep image tensor as is (don't add batch dimension for single sample)
                    cond_batch[key] = value
                else:
                    cond_batch[key] = value
            
            return target_batch, cond_batch
        else:
            # Just target image - add batch dimension
            return sample.unsqueeze(0)
    else:
        # Multiple samples - use default collation with proper handling
        targets = []
        text_prompts = []
        images = []
        
        for sample in batch:
            if isinstance(sample, tuple):
                target, cond_inputs = sample
                targets.append(target)
                text_prompts.append(cond_inputs.get('text', ''))
                images.append(cond_inputs.get('image'))
            else:
                targets.append(sample)
                text_prompts.append('')
                images.append(None)
        
        # Stack targets
        target_batch = torch.stack(targets, dim=0)
        
        # Create conditioning batch
        cond_batch = {
            'text': text_prompts,
            'image': torch.stack([img for img in images if img is not None], dim=0) if any(img is not None for img in images) else None
        }
        
        return target_batch, cond_batch

def test_collate_fix():
    """Test the fixed collate function"""
    print("Testing collate function fix...")
    
    # Load config
    config_path = 'config/sar_config.yaml'
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        return
    
    # Create dataset
    dataset_config = config['dataset_params']
    condition_config = config['ldm_params']['condition_config']
    
    try:
        test_dataset = SARDataset(
            split='test',
            im_path=dataset_config['im_path'],
            im_size=dataset_config['im_size'],
            im_channels=dataset_config['im_channels'],
            condition_config=condition_config,
            train_split=dataset_config['train_split'],
            val_split=dataset_config['val_split'],
            test_split=dataset_config['test_split'],
            use_latents=False
        )
        
        print(f"Dataset loaded: {len(test_dataset)} samples")
        
        # Test with a small subset
        subset_indices = list(range(min(5, len(test_dataset))))
        test_subset = torch.utils.data.Subset(test_dataset, subset_indices)
        
        # Create DataLoader with fixed collate function
        test_loader = DataLoader(test_subset, batch_size=1, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
        
        print("\n=== Testing DataLoader with fixed collate function ===")
        for i, data in enumerate(test_loader):
            print(f"\nBatch {i}:")
            print(f"  Data type: {type(data)}")
            
            if isinstance(data, tuple):
                ground_truth, cond_inputs = data
                print(f"  Ground truth type: {type(ground_truth)}, shape: {ground_truth.shape}")
                print(f"  Cond inputs type: {type(cond_inputs)}")
                print(f"  Cond inputs keys: {cond_inputs.keys()}")
                
                if 'text' in cond_inputs:
                    text_prompt = cond_inputs['text']
                    print(f"    text type: {type(text_prompt)}")
                    if isinstance(text_prompt, str):
                        print(f"    text content (first 50 chars): {text_prompt[:50]}...")
                    else:
                        print(f"    text content: {text_prompt}")
                
                if 'image' in cond_inputs:
                    sar_image = cond_inputs['image']
                    print(f"    image type: {type(sar_image)}")
                    if isinstance(sar_image, torch.Tensor):
                        print(f"    image shape: {sar_image.shape}")
                    else:
                        print(f"    ERROR: image is not a tensor!")
                        
            else:
                print(f"  Single tensor: type={type(data)}, shape={data.shape if hasattr(data, 'shape') else 'No shape'}")
            
            if i >= 2:  # Test first 3 samples
                break
                
        print("\n✓ Collate function test completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_collate_fix()
