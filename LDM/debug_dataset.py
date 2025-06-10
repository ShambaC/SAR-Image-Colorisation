#!/usr/bin/env python3
"""
Debug script to check SAR dataset structure
"""

import yaml
import torch
from torch.utils.data import DataLoader
from dataset.sar_dataset import SARDataset

def debug_dataset():
    # Load configuration
    with open('config/sar_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    dataset_config = config['dataset_params']
    condition_config = config['ldm_params']['condition_config']
    
    # Create test dataset
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
    
    print(f"Dataset length: {len(test_dataset)}")
    
    # Test single item access
    if len(test_dataset) > 0:
        print("\n=== Testing single item access ===")
        item = test_dataset[0]
        print(f"Item type: {type(item)}")
        
        if isinstance(item, tuple):
            target, cond_inputs = item
            print(f"Target type: {type(target)}, shape: {target.shape}")
            print(f"Cond inputs type: {type(cond_inputs)}")
            print(f"Cond inputs keys: {cond_inputs.keys()}")
            
            for key, value in cond_inputs.items():
                print(f"  {key}: type={type(value)}, shape={value.shape if hasattr(value, 'shape') else 'N/A'}")
        else:
            print(f"Single tensor: type={type(item)}, shape={item.shape}")
    
    # Test DataLoader without custom collate
    print("\n=== Testing DataLoader without custom collate ===")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    for i, data in enumerate(test_loader):
        print(f"Batch {i}: type={type(data)}")
        
        if isinstance(data, tuple):
            target, cond_inputs = data
            print(f"  Target: type={type(target)}, shape={target.shape}")
            print(f"  Cond inputs: type={type(cond_inputs)}")
            print(f"  Cond inputs keys: {cond_inputs.keys()}")
            
            for key, value in cond_inputs.items():
                print(f"    {key}: type={type(value)}")
                if hasattr(value, 'shape'):
                    print(f"         shape={value.shape}")
                elif isinstance(value, list):
                    print(f"         list length={len(value)}, first item type={type(value[0]) if value else 'empty'}")
                else:
                    print(f"         value={value}")
        else:
            print(f"  Single tensor: type={type(data)}, shape={data.shape}")
        
        if i >= 2:  # Only test first 3 items
            break
    
    print("\nDebug complete!")

if __name__ == '__main__':
    debug_dataset()
