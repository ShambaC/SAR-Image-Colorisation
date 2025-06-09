#!/usr/bin/env python3
"""
Validation Script for SAR Image Colorization System
Tests the key components without running full training
"""

import torch
import sys
import os

def test_dataset_loading():
    """Test dataset loading with custom collate function"""
    print("Testing dataset loading...")
    try:
        from dataset import create_dataloaders
        
        # Test with small batch and no workers to avoid multiprocessing issues
        train_loader, test_loader = create_dataloaders(
            dataset_path="../Dataset",
            batch_size=2,
            num_workers=0,  # Avoid multiprocessing for testing
            train_split=0.8
        )
        
        print(f"✅ Dataset loading successful")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Test batches: {len(test_loader)}")
        
        # Test loading one batch
        for batch in train_loader:
            print(f"   Batch structure:")
            print(f"     - s1_image shape: {batch['s1_image'].shape}")
            print(f"     - s2_image shape: {batch['s2_image'].shape}")
            print(f"     - prompt type: {type(batch['prompt'])}, count: {len(batch['prompt'])}")
            print(f"     - metadata type: {type(batch['metadata'])}, count: {len(batch['metadata'])}")
            print(f"     - sample prompt: {batch['prompt'][0][:50]}...") 
            break
            
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False

def test_model_components():
    """Test model component initialization"""
    print("\nTesting model components...")
    try:
        from encoder import VAE_Encoder
        from decoder import VAE_Decoder
        from diffusion import Diffusion
        from clip import CLIP
        from tokenizer import SimpleTokenizer
        
        # Create tokenizer
        tokenizer = SimpleTokenizer()
        
        # Test VAE Encoder with noise parameter
        encoder = VAE_Encoder()
        test_images = torch.randn(2, 3, 256, 256)  # Batch of 2 images
        test_noise = torch.randn(2, 4, 32, 32)     # Noise for latent space
        
        encoded = encoder(test_images, test_noise)
        print(f"✅ VAE Encoder working with noise parameter")
        print(f"   Input shape: {test_images.shape}")
        print(f"   Noise shape: {test_noise.shape}")
        print(f"   Output shape: {encoded.shape}")
        
        # Test other components
        decoder = VAE_Decoder()
        diffusion = Diffusion()
        clip_model = CLIP(
            n_vocab=tokenizer.get_vocab_size(),
            n_embd=768,
            n_token=77,
            n_head=12,
            n_layers=12
        )
        
        print(f"✅ All model components initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Model component test failed: {e}")
        return False

def test_training_model_class():
    """Test the training model class initialization"""
    print("\nTesting training model class...")
    try:
        from train_model import SARColorizer
        from tokenizer import SimpleTokenizer
        
        tokenizer = SimpleTokenizer()
        clip_config = {
            "n_embd": 768,
            "max_length": 77,
            "n_head": 12,
            "n_layers": 12
        }
        
        model = SARColorizer(tokenizer, clip_config)
        
        # Test encode_images method with noise parameter
        test_images = torch.randn(1, 3, 256, 256)
        encoded = model.encode_images(test_images)
        
        print(f"✅ SARColorizer model initialized successfully")
        print(f"   encode_images working with automatic noise generation")
        print(f"   Input shape: {test_images.shape}")
        print(f"   Output shape: {encoded.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training model class test failed: {e}")
        return False

def test_configurations():
    """Test configuration files"""
    print("\nTesting configuration files...")
    try:
        import json
        
        # Test CLIP config
        if os.path.exists("configs/clip_config.json"):
            with open("configs/clip_config.json", 'r') as f:
                clip_config = json.load(f)
            print(f"✅ CLIP config loaded: {len(clip_config)} parameters")
        else:
            print("⚠️  CLIP config not found, will use defaults")
        
        # Test model config
        if os.path.exists("configs/model_config.json"):
            with open("configs/model_config.json", 'r') as f:
                model_config = json.load(f)
            print(f"✅ Model config loaded: {len(model_config)} parameters")
        else:
            print("⚠️  Model config not found, will use defaults")
            
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print("🧪 SAR Image Colorization System Validation")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Test 1: Dataset loading with custom collate function
    if not test_dataset_loading():
        all_tests_passed = False
    
    # Test 2: Model components
    if not test_model_components():
        all_tests_passed = False
    
    # Test 3: Training model class
    if not test_training_model_class():
        all_tests_passed = False
    
    # Test 4: Configuration files
    if not test_configurations():
        all_tests_passed = False
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 All validation tests passed!")
        print("✅ System is ready for training")
        print("\nNext steps:")
        print("1. Transfer to your GPU system")
        print("2. Run: python quick_train.py --dataset_path ../Dataset")
        print("3. Monitor with tensorboard")
    else:
        print("❌ Some tests failed - check the errors above")
        print("🔧 Fix any issues before proceeding to training")

if __name__ == "__main__":
    main()
