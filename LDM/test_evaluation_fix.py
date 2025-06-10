#!/usr/bin/env python3
"""
Quick test script to verify the batch size fix
"""

import sys
import os
import yaml
import torch

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluate_model import evaluate_model

def test_evaluation_fix():
    """Test evaluation with a few samples"""
    print("Testing evaluation with batch size fix...")
    
    # Load config
    config_path = 'config/sar_config.yaml'
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        return
    
    try:
        # Run evaluation on just 5 samples
        print("Running evaluation on 5 samples to test fix...")
        results, results_dir = evaluate_model(
            config, 
            num_samples=5,  # Test with just 5 samples
            guidance_scale=7.5,
            save_samples=True
        )
        
        print(f"✓ Test evaluation completed successfully!")
        print(f"✓ Processed {len(results['sample_info'])} samples")
        print(f"✓ Results saved in: {results_dir}")
        
        # Print sample metrics
        if results['sample_info']:
            print(f"Sample metrics:")
            for sample in results['sample_info'][:3]:  # Show first 3
                print(f"  Sample {sample['sample_id']}: PSNR={sample['psnr']:.3f}, SSIM={sample['ssim']:.3f}")
        
    except Exception as e:
        print(f"✗ Test evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_evaluation_fix()
