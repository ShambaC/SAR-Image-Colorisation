"""
Quick Training Script for SAR Image Colorization
This script demonstrates the complete training workflow
"""
import argparse
import json
import os
import sys
from pathlib import Path

def check_prerequisites():
    """Check if all prerequisites are met"""
    print("Checking prerequisites...")
    
    # Check dataset
    dataset_path = "../Dataset"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}")
        print("Please ensure the dataset is in the correct location.")
        return False
    
    # Check if configs exist
    os.makedirs("configs", exist_ok=True)
    
    clip_config = "configs/clip_config.json"
    if not os.path.exists(clip_config):
        print(f"⚠️  CLIP config not found, will use defaults")
    
    model_config = "configs/model_config.json"
    if not os.path.exists(model_config):
        print(f"⚠️  Model config not found, will use defaults")
    
    print("✅ Prerequisites check completed")
    return True

def train_clip_model(args):
    """Train the CLIP model"""
    print("\n" + "="*50)
    print("STEP 1: Training CLIP Model")
    print("="*50)
    
    clip_cmd = [
        "python", "train_clip.py",
        "--dataset_path", args.dataset_path,
        "--save_dir", args.clip_save_dir,
        "--log_dir", args.clip_log_dir
    ]
    
    # If config file exists, use only config file parameters
    if args.clip_config and os.path.exists(args.clip_config):
        clip_cmd.extend(["--config_file", args.clip_config])
        print(f"Using configuration from: {args.clip_config}")
    else:
        # Use command line parameters only if no config file
        clip_cmd.extend([
            "--batch_size", str(args.clip_batch_size),
            "--epochs", str(args.clip_epochs),
            "--learning_rate", str(args.clip_lr)
        ])
        print("Using command line parameters (no config file found)")
    
    print(f"Running: {' '.join(clip_cmd)}")
    
    if args.dry_run:
        print("(Dry run - command not executed)")
        return True
    
    import subprocess
    try:
        result = subprocess.run(clip_cmd, check=True, capture_output=True, text=True)
        print("✅ CLIP training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ CLIP training failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def train_main_model(args):
    """Train the main diffusion model"""
    print("\n" + "="*50)
    print("STEP 2: Training Main Diffusion Model")
    print("="*50)
    
    # Find the best CLIP checkpoint
    clip_checkpoint = os.path.join(args.clip_save_dir, "best_clip.pt")
    if not os.path.exists(clip_checkpoint):
        clip_checkpoint = os.path.join(args.clip_save_dir, "latest_clip.pt")
    
    if not os.path.exists(clip_checkpoint):
        print(f"❌ No CLIP checkpoint found in {args.clip_save_dir}")
        return False
    
    main_cmd = [
        "python", "train_model.py",
        "--dataset_path", args.dataset_path,
        "--clip_checkpoint", clip_checkpoint,
        "--save_dir", args.model_save_dir,
        "--log_dir", args.model_log_dir
    ]
    
    # If config file exists, use only config file parameters
    if args.model_config and os.path.exists(args.model_config):
        main_cmd.extend(["--config", args.model_config])
        print(f"Using configuration from: {args.model_config}")
    else:
        # Use command line parameters only if no config file
        main_cmd.extend([
            "--batch_size", str(args.model_batch_size),
            "--epochs", str(args.model_epochs),
            "--learning_rate", str(args.model_lr),
            "--gradient_accumulation_steps", str(args.grad_accumulation)
        ])
        if args.mixed_precision:
            main_cmd.append("--mixed_precision")
        print("Using command line parameters (no config file found)")
    
    print(f"Running: {' '.join(main_cmd)}")
    
    if args.dry_run:
        print("(Dry run - command not executed)")
        return True
    
    import subprocess
    try:
        result = subprocess.run(main_cmd, check=True, capture_output=True, text=True)
        print("✅ Main model training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Main model training failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def run_inference_test(args):
    """Run a quick inference test"""
    print("\n" + "="*50)
    print("STEP 3: Testing Inference")
    print("="*50)
    
    # Find model checkpoints
    model_checkpoint = os.path.join(args.model_save_dir, "best_model.pt")
    if not os.path.exists(model_checkpoint):
        model_checkpoint = os.path.join(args.model_save_dir, "latest_model.pt")
    
    clip_checkpoint = os.path.join(args.clip_save_dir, "best_clip.pt")
    if not os.path.exists(clip_checkpoint):
        clip_checkpoint = os.path.join(args.clip_save_dir, "latest_clip.pt")
    
    if not os.path.exists(model_checkpoint) or not os.path.exists(clip_checkpoint):
        print("❌ Required checkpoints not found for inference test")
        return False
    
    # Create test output directory
    test_output_dir = "quick_test_results"
    os.makedirs(test_output_dir, exist_ok=True)
    
    inference_cmd = [
        "python", "infer_model.py",
        "--mode", "dataset",
        "--model_checkpoint", model_checkpoint,
        "--clip_checkpoint", clip_checkpoint,
        "--dataset_path", args.dataset_path,
        "--output_dir", test_output_dir,
        "--num_samples", "5",
        "--guidance_scale", "7.5",
        "--num_inference_steps", "20"  # Faster for testing
    ]
    
    print(f"Running: {' '.join(inference_cmd)}")
    
    if args.dry_run:
        print("(Dry run - command not executed)")
        return True
    
    import subprocess
    try:
        result = subprocess.run(inference_cmd, check=True, capture_output=True, text=True)
        print(f"✅ Inference test completed successfully")
        print(f"Test results saved to {test_output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Inference test failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Quick Training Script for SAR Colorization")
    
    # General arguments
    parser.add_argument("--dataset_path", type=str, default="../Dataset",
                       help="Path to the dataset")
    parser.add_argument("--dry_run", action="store_true",
                       help="Show commands without executing them")
    parser.add_argument("--skip_clip", action="store_true",
                       help="Skip CLIP training (use existing checkpoint)")
    parser.add_argument("--skip_main", action="store_true",
                       help="Skip main model training")
    parser.add_argument("--skip_inference", action="store_true",
                       help="Skip inference test")
    
    # CLIP training arguments
    parser.add_argument("--clip_config", type=str, default="configs/clip_config.json",
                       help="CLIP configuration file")
    parser.add_argument("--clip_batch_size", type=int, default=16,
                       help="CLIP training batch size")
    parser.add_argument("--clip_epochs", type=int, default=20,
                       help="CLIP training epochs")
    parser.add_argument("--clip_lr", type=float, default=1e-4,
                       help="CLIP learning rate")
    parser.add_argument("--clip_save_dir", type=str, default="./checkpoints/clip",
                       help="CLIP checkpoint save directory")
    parser.add_argument("--clip_log_dir", type=str, default="./logs/clip",
                       help="CLIP log directory")
    
    # Main model training arguments
    parser.add_argument("--model_config", type=str, default="configs/model_config.json",
                       help="Main model configuration file")
    parser.add_argument("--model_batch_size", type=int, default=2,
                       help="Main model training batch size")
    parser.add_argument("--model_epochs", type=int, default=50,
                       help="Main model training epochs")
    parser.add_argument("--model_lr", type=float, default=1e-4,
                       help="Main model learning rate")
    parser.add_argument("--model_save_dir", type=str, default="./checkpoints/model",
                       help="Main model checkpoint save directory")
    parser.add_argument("--model_log_dir", type=str, default="./logs/model",
                       help="Main model log directory")
    parser.add_argument("--mixed_precision", action="store_true",
                       help="Use mixed precision training")
    parser.add_argument("--grad_accumulation", type=int, default=4,
                       help="Gradient accumulation steps")
    
    args = parser.parse_args()
    
    print("🚀 SAR Image Colorization - Quick Training Script")
    print("="*60)
    
    # Check prerequisites
    if not check_prerequisites():
        return
    
    # Create directories
    os.makedirs(args.clip_save_dir, exist_ok=True)
    os.makedirs(args.clip_log_dir, exist_ok=True)
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.model_log_dir, exist_ok=True)
    
    success = True
    
    # Step 1: Train CLIP model
    if not args.skip_clip:
        success = train_clip_model(args)
        if not success:
            print("❌ Training failed at CLIP stage")
            return
    else:
        print("⏭️  Skipping CLIP training")
    
    # Step 2: Train main model
    if success and not args.skip_main:
        success = train_main_model(args)
        if not success:
            print("❌ Training failed at main model stage")
            return
    else:
        print("⏭️  Skipping main model training")
    
    # Step 3: Test inference
    if success and not args.skip_inference:
        success = run_inference_test(args)
        if not success:
            print("❌ Inference test failed")
            return
    else:
        print("⏭️  Skipping inference test")
    
    if success:
        print("\n🎉 Training pipeline completed successfully!")
        print("Next steps:")
        print("1. Check tensorboard logs for training progress:")
        print(f"   tensorboard --logdir {args.clip_log_dir}")
        print(f"   tensorboard --logdir {args.model_log_dir}")
        print("2. Run inference on your own images:")
        print("   python infer_model.py --mode single --input_image <path> --region <region> --season <season>")
        print("3. Fine-tune hyperparameters if needed")
    else:
        print("\n❌ Training pipeline failed. Check the error messages above.")

if __name__ == "__main__":
    main()
