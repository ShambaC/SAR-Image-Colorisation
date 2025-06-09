#!/usr/bin/env python3
"""
Test script for timestep embedding fix
"""

import torch
import sys
import os

def test_timestep_embedding():
    """Test the timestep embedding function"""
    print("Testing timestep embedding function...")
    
    try:
        # Import our function
        sys.path.append('.')
        from train_model import get_timestep_embedding
        
        # Test with different batch sizes and timesteps
        batch_size = 4
        timesteps = torch.randint(0, 1000, (batch_size,))
        
        print(f"Input timesteps: {timesteps}")
        print(f"Input shape: {timesteps.shape}")
        print(f"Input dtype: {timesteps.dtype}")
        
        # Convert to embeddings
        embeddings = get_timestep_embedding(timesteps, 320)
        
        print(f"Output shape: {embeddings.shape}")
        print(f"Output dtype: {embeddings.dtype}")
        print(f"Expected shape: ({batch_size}, 320)")
        
        # Verify output
        assert embeddings.shape == (batch_size, 320), f"Expected shape ({batch_size}, 320), got {embeddings.shape}"
        assert embeddings.dtype == torch.float32, f"Expected float32, got {embeddings.dtype}"
        
        print("✅ Timestep embedding function working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Timestep embedding test failed: {e}")
        return False

def test_model_forward():
    """Test the model forward pass with timestep embeddings"""
    print("\nTesting model forward pass...")
    
    try:
        from train_model import SARColorizationModel, get_timestep_embedding
        from tokenizer import SimpleTokenizer
        
        # Create simple configs
        clip_config = {
            "n_embd": 768,
            "max_length": 77,
            "n_head": 12,
            "n_layers": 12
        }
        
        diffusion_config = {}
        
        # Create tokenizer
        tokenizer = SimpleTokenizer()
        
        # Create model
        model = SARColorizationModel(clip_config, diffusion_config, tokenizer)
        
        # Create test inputs
        batch_size = 2
        sar_images = torch.randn(batch_size, 3, 256, 256)
        prompts = ["Colorise image, Region: arctic, Season: winter"] * batch_size
        
        # Test forward pass
        with torch.no_grad():
            predicted_noise, target_noise = model(sar_images, prompts)
        
        print(f"Forward pass successful!")
        print(f"Predicted noise shape: {predicted_noise.shape}")
        print(f"Target noise shape: {target_noise.shape}")
        
        print("✅ Model forward pass working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Model forward test failed: {e}")
        return False

def main():
    """Run timestep embedding tests"""
    print("🧪 Testing Timestep Embedding Fix")
    print("=" * 40)
    
    success = True
    
    # Test 1: Timestep embedding function
    if not test_timestep_embedding():
        success = False
    
    # Test 2: Model forward pass with embeddings
    if not test_model_forward():
        success = False
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 All timestep embedding tests passed!")
        print("✅ Training should now work without dtype errors")
    else:
        print("❌ Some tests failed - check the errors above")

if __name__ == "__main__":
    main()
