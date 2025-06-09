"""
Test script to validate dataset loading and basic functionality
"""
import os
import sys
import torch
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import SARDataset, create_dataloaders, get_dataset_statistics
from tokenizer import SimpleTokenizer, create_tokenizer_from_dataset

def test_dataset_loading():
    """Test basic dataset loading"""
    print("=" * 50)
    print("Testing Dataset Loading")
    print("=" * 50)
    
    dataset_path = "../Dataset"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path {dataset_path} does not exist!")
        print("Please make sure the dataset is in the correct location.")
        return False
    
    try:
        # Get dataset statistics
        print("Getting dataset statistics...")
        stats = get_dataset_statistics(dataset_path)
        
        print(f"✅ Dataset Statistics:")
        print(f"   Total samples: {stats['total_samples']}")
        print(f"   Number of CSV files: {stats['num_csv_files']}")
        print(f"   Regions: {stats['regions']}")
        print(f"   Seasons: {stats['seasons']}")
        print(f"   Countries: {len(stats['countries'])} countries")
        
        # Test dataset loading
        print("\nTesting dataset loading...")
        dataset = SARDataset(dataset_path, train=True, train_split=0.8)
        print(f"✅ Train dataset loaded: {len(dataset)} samples")
        
        test_dataset = SARDataset(dataset_path, train=False, train_split=0.8)
        print(f"✅ Test dataset loaded: {len(test_dataset)} samples")
        
        # Test loading a sample
        if len(dataset) > 0:
            print("\nTesting sample loading...")
            sample = dataset[0]
            print(f"✅ Sample loaded successfully:")
            print(f"   S1 image shape: {sample['s1_image'].shape}")
            print(f"   S2 image shape: {sample['s2_image'].shape}")
            print(f"   Prompt: {sample['prompt']}")
            print(f"   Region: {sample['metadata']['region']}")
            print(f"   Season: {sample['metadata']['season']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

def test_tokenizer():
    """Test tokenizer functionality"""
    print("\n" + "=" * 50)
    print("Testing Tokenizer")
    print("=" * 50)
    
    try:
        # Test basic tokenizer
        print("Testing basic tokenizer...")
        tokenizer = SimpleTokenizer(vocab_size=100, max_length=77)
        
        test_prompts = [
            "Colorise image, Region: tropical, Season: summer",
            "Colorise image, Region: arctic, Season: winter",
            "Colorise image, Region: temperate, Season: spring"
        ]
        
        for prompt in test_prompts:
            encoded = tokenizer.encode(prompt)
            decoded = tokenizer.decode(encoded)
            print(f"✅ Prompt: {prompt}")
            print(f"   Encoded length: {len(encoded)}")
            print(f"   Decoded: {decoded}")
        
        # Test batch encoding
        batch_encoded = tokenizer.encode_batch(test_prompts)
        print(f"✅ Batch encoding shape: {batch_encoded.shape}")
        
        # Test dataset-based tokenizer
        dataset_path = "../Dataset"
        if os.path.exists(dataset_path):
            print("\nTesting dataset-based tokenizer...")
            dataset_tokenizer = create_tokenizer_from_dataset(dataset_path, vocab_size=1000)
            print(f"✅ Dataset tokenizer created with vocab size: {dataset_tokenizer.get_vocab_size()}")
            
            # Test on sample prompts
            for prompt in test_prompts:
                encoded = dataset_tokenizer.encode(prompt)
                decoded = dataset_tokenizer.decode(encoded)
                print(f"✅ Dataset tokenizer - Prompt: {prompt}")
                print(f"   Decoded: {decoded}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing tokenizer: {e}")
        return False

def test_model_imports():
    """Test that all model components can be imported"""
    print("\n" + "=" * 50)
    print("Testing Model Imports")
    print("=" * 50)
    
    try:
        print("Testing CLIP import...")
        from clip import CLIP, CLIPEmbedding, CLIPLayer
        print("✅ CLIP components imported successfully")
        
        print("Testing attention import...")
        from attention import SelfAttention, CrossAttention
        print("✅ Attention components imported successfully")
        
        print("Testing diffusion import...")
        from diffusion import Diffusion, TimeEmbedding, UNET_ResidualBlock
        print("✅ Diffusion components imported successfully")
        
        print("Testing VAE import...")
        from encoder import VAE_Encoder
        from decoder import VAE_Decoder
        print("✅ VAE components imported successfully")
        
        # Test model creation
        print("\nTesting model creation...")
        clip_model = CLIP(n_vocab=1000, n_embd=768, n_token=77, n_head=12, n_layers=12)
        print(f"✅ CLIP model created: {sum(p.numel() for p in clip_model.parameters()):,} parameters")
        
        vae_encoder = VAE_Encoder()
        vae_decoder = VAE_Decoder()
        print("✅ VAE models created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error importing models: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_setup():
    """Test training script imports and basic setup"""
    print("\n" + "=" * 50)
    print("Testing Training Setup")
    print("=" * 50)
    
    try:
        print("Testing training imports...")
        from train_clip import CLIPTrainer
        from train_model import SARColorizationModel
        print("✅ Training components imported successfully")
        
        # Test configuration loading
        config_path = "configs/clip_config.json"
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"✅ CLIP configuration loaded from {config_path}")
        else:
            print(f"⚠️  CLIP configuration file not found at {config_path}")
        
        model_config_path = "configs/model_config.json"
        if os.path.exists(model_config_path):
            import json
            with open(model_config_path, 'r') as f:
                config = json.load(f)
            print(f"✅ Model configuration loaded from {model_config_path}")
        else:
            print(f"⚠️  Model configuration file not found at {model_config_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing training setup: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 SAR Image Colorization - System Test")
    print("=" * 50)
    
    # Check Python and PyTorch versions
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print()
    
    tests = [
        ("Dataset Loading", test_dataset_loading),
        ("Tokenizer", test_tokenizer),
        ("Model Imports", test_model_imports),
        ("Training Setup", test_training_setup),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Your system is ready for training.")
    else:
        print("⚠️  Some tests failed. Please fix the issues before proceeding.")
        print("\nNext steps:")
        print("1. Check that the dataset is in the correct location (../Dataset)")
        print("2. Ensure all dependencies are installed (pip install -r requirements.txt)")
        print("3. Check for any missing files or configuration issues")

if __name__ == "__main__":
    main()
