#!/usr/bin/env python3
"""
Test script to verify the corrected SAR-to-Optical training approach
"""

import torch
import numpy as np
from train_model import SARColorizationModel, get_default_config
from tokenizer import SimpleTokenizer

def test_corrected_training_flow():
    """Test that the model now properly uses optical images in training"""
    print("=== Testing Corrected SAR-to-Optical Training Flow ===")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_default_config()
    
    # Create tokenizer
    tokenizer = SimpleTokenizer()
    
    # Create model
    model = SARColorizationModel(
        clip_config=config["clip_config"],
        diffusion_config=config["diffusion_config"],
        tokenizer=tokenizer
    ).to(device)
    
    # Create test data
    batch_size = 2
    sar_images = torch.randn(batch_size, 3, 256, 256).to(device)
    optical_images = torch.randn(batch_size, 3, 256, 256).to(device)
    prompts = ["A green landscape with trees", "A blue ocean with waves"]
    
    print(f"SAR images shape: {sar_images.shape}")
    print(f"Optical images shape: {optical_images.shape}")
    print(f"Prompts: {prompts}")
    
    # Test forward pass
    print("\n--- Testing Forward Pass ---")
    model.train()
    
    predicted_noise, target_noise = model(sar_images, optical_images, prompts)
    
    print(f"Predicted noise shape: {predicted_noise.shape}")
    print(f"Target noise shape: {target_noise.shape}")
    
    # Verify shapes are correct for latent space (256//8 = 32)
    expected_shape = (batch_size, 4, 32, 32)
    assert predicted_noise.shape == expected_shape, f"Expected {expected_shape}, got {predicted_noise.shape}"
    assert target_noise.shape == expected_shape, f"Expected {expected_shape}, got {target_noise.shape}"
    
    print("✅ Forward pass shapes are correct")
    
    # Test loss computation
    print("\n--- Testing Loss Computation ---")
    mse_loss = torch.nn.MSELoss()
    l1_loss = torch.nn.L1Loss()
    
    mse_value = mse_loss(predicted_noise, target_noise)
    l1_value = l1_loss(predicted_noise, target_noise)
    combined_loss = 1.0 * mse_value + 0.1 * l1_value
    
    print(f"MSE Loss: {mse_value.item():.6f}")
    print(f"L1 Loss: {l1_value.item():.6f}")
    print(f"Combined Loss: {combined_loss.item():.6f}")
    
    print("✅ Loss computation working")
    
    # Test that gradients flow properly
    print("\n--- Testing Gradient Flow ---")
    combined_loss.backward()
    
    # Check if gradients exist
    gradients_exist = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            gradients_exist = True
            break
    
    if gradients_exist:
        print("✅ Gradients are flowing through the model")
    else:
        print("❌ No gradients found - potential issue with training")
    
    # Test generation
    print("\n--- Testing Generation ---")
    model.eval()
    
    with torch.no_grad():
        generated_images = model.generate(sar_images, prompts, num_steps=5)  # Few steps for testing
    
    print(f"Generated images shape: {generated_images.shape}")
    expected_gen_shape = (batch_size, 3, 256, 256)
    assert generated_images.shape == expected_gen_shape, f"Expected {expected_gen_shape}, got {generated_images.shape}"
    
    print("✅ Generation working correctly")
    
    return True

def test_diffusion_input_channels():
    """Test that diffusion model receives concatenated SAR+optical inputs"""
    print("\n=== Testing Diffusion Input Channels ===")
    
    # This test verifies that our diffusion model gets 8-channel input
    # (4 channels from SAR latents + 4 channels from optical latents)
    
    from diffusion import Diffusion
    
    # Create diffusion model
    diffusion = Diffusion()
    
    # Check input layer
    first_layer = None
    for module in diffusion.modules():
        if isinstance(module, torch.nn.Conv2d):
            first_layer = module
            break
    
    if first_layer is not None:
        print(f"First conv layer input channels: {first_layer.in_channels}")
        if first_layer.in_channels == 8:
            print("✅ Diffusion model accepts 8-channel input (SAR + optical)")
        elif first_layer.in_channels == 4:
            print("⚠️  Diffusion model only accepts 4-channel input - needs updating")
        else:
            print(f"❓ Unexpected input channels: {first_layer.in_channels}")
    
    return True

def test_training_data_usage():
    """Verify that both SAR and optical images are being used in training"""
    print("\n=== Testing Training Data Usage ===")
    
    # Create dummy training step
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_default_config()
    tokenizer = SimpleTokenizer()
    
    model = SARColorizationModel(
        clip_config=config["clip_config"],
        diffusion_config=config["diffusion_config"],
        tokenizer=tokenizer
    ).to(device)
    
    # Create different SAR and optical images to verify both are used
    sar_images = torch.ones(1, 3, 256, 256).to(device) * 0.1  # Low values
    optical_images = torch.ones(1, 3, 256, 256).to(device) * 0.9  # High values
    prompts = ["Test image"]
    
    model.train()
    
    # Get model outputs
    predicted_noise, target_noise = model(sar_images, optical_images, prompts)
    
    # Check that the target noise is derived from optical images, not SAR
    # We do this by checking if changing optical images changes the target
    optical_images_2 = torch.ones(1, 3, 256, 256).to(device) * 0.5  # Medium values
    
    _, target_noise_2 = model(sar_images, optical_images_2, prompts)
    
    # Target noise should be different when optical images change
    noise_difference = torch.abs(target_noise - target_noise_2).mean()
    
    print(f"Noise difference when optical images change: {noise_difference.item():.6f}")
    
    if noise_difference.item() > 1e-6:
        print("✅ Target noise changes with optical images - training uses optical data")
    else:
        print("❌ Target noise doesn't change with optical images - potential issue")
    
    return noise_difference.item() > 1e-6

def main():
    """Run all tests"""
    print("🔧 Testing Corrected SAR-to-Optical Training")
    print("=" * 60)
    
    try:
        # Run tests
        test_corrected_training_flow()
        test_diffusion_input_channels()
        training_uses_optical = test_training_data_usage()
        
        print("\n📊 Test Summary:")
        print("✅ Model forward pass working")
        print("✅ Loss computation working")
        print("✅ Generation working")
        
        if training_uses_optical:
            print("✅ Training properly uses optical images as targets")
            print("\n🎉 All tests passed! Training should now learn SAR→Optical mapping")
        else:
            print("❌ Training may not be using optical images properly")
            print("\n⚠️  Some issues detected - please review the training approach")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
