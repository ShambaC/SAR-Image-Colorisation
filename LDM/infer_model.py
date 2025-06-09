import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import argparse
import json
import os
from pathlib import Path
import torchvision.transforms as transforms

from dataset import SARDataset
from tokenizer import SimpleTokenizer
from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion
from ddpm import DDPMSampler
from pipeline import generate


class SARInference:
    """Inference pipeline for SAR image colorization"""
    
    def __init__(
        self,
        model_checkpoint_path: str,
        clip_checkpoint_path: str = None,
        device: str = "auto"
    ):
        """
        Args:
            model_checkpoint_path: Path to the trained diffusion model checkpoint
            clip_checkpoint_path: Path to the trained CLIP model checkpoint
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model configuration and checkpoints
        self.config = self._load_config(model_checkpoint_path)
        self.models = self._load_models(model_checkpoint_path, clip_checkpoint_path)
        self.tokenizer = self._load_tokenizer()
        
        # Setup image transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.denormalize = transforms.Normalize(
            mean=[-1, -1, -1], 
            std=[2, 2, 2]
        )
    
    def _load_config(self, checkpoint_path: str) -> dict:
        """Load model configuration"""
        config_path = Path(checkpoint_path).parent / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                "clip_config": {
                    "n_vocab": 1000,
                    "n_embd": 768,
                    "max_length": 77,
                    "n_head": 12,
                    "n_layers": 12
                },
                "diffusion_config": {
                    "in_channels": 8,  # 4 (SAR latent) + 4 (noise)
                    "out_channels": 4,
                    "model_channels": 320,
                    "attention_resolutions": [4, 2, 1],
                    "num_res_blocks": 2,
                    "channel_mult": [1, 2, 4, 4],
                    "num_heads": 8,
                    "use_spatial_transformer": True,
                    "context_dim": 768
                }
            }
    def _load_tokenizer(self) -> SimpleTokenizer:
        """Load or create tokenizer"""
        tokenizer_path = Path(self.config.get("tokenizer_path", "tokenizer.json"))
        if tokenizer_path.exists():
            tokenizer = SimpleTokenizer()
            tokenizer.load_vocab(str(tokenizer_path))
        else:
            # Create default tokenizer
            tokenizer = SimpleTokenizer(vocab_size=1000, max_length=77)
        return tokenizer
    
    def _load_models(self, model_checkpoint_path: str, clip_checkpoint_path: str = None) -> dict:
        """Load all required models"""
        models = {}
        
        # Load CLIP
        clip_config = self.config["clip_config"]
        models["clip"] = CLIP(
            n_vocab=clip_config["n_vocab"],
            n_embd=clip_config["n_embd"],
            n_token=clip_config["max_length"],
            n_head=clip_config["n_head"],
            n_layers=clip_config["n_layers"]
        )
        
        if clip_checkpoint_path and os.path.exists(clip_checkpoint_path):
            clip_state = torch.load(clip_checkpoint_path, map_location=self.device)
            models["clip"].load_state_dict(clip_state["model_state_dict"])
            print(f"Loaded CLIP from {clip_checkpoint_path}")
        
        # Load VAE Encoder/Decoder
        models["encoder"] = VAE_Encoder()
        models["decoder"] = VAE_Decoder()
          # Load Diffusion model
        models["diffusion"] = Diffusion()  # Diffusion class takes no parameters
        
        # Load main model checkpoint
        if os.path.exists(model_checkpoint_path):
            checkpoint = torch.load(model_checkpoint_path, map_location=self.device)
            
            # Load individual model states
            if "encoder_state_dict" in checkpoint:
                models["encoder"].load_state_dict(checkpoint["encoder_state_dict"])
            if "decoder_state_dict" in checkpoint:
                models["decoder"].load_state_dict(checkpoint["decoder_state_dict"])
            if "diffusion_state_dict" in checkpoint:
                models["diffusion"].load_state_dict(checkpoint["diffusion_state_dict"])
            
            print(f"Loaded model checkpoint from {model_checkpoint_path}")
        
        # Move models to device and set to eval mode
        for model in models.values():
            model.to(self.device)
            model.eval()
        
        return models
    
    def generate_prompt(self, region: str, season: str) -> str:
        """Generate text prompt for given region and season"""
        return f"Colorise image, Region: {region}, Season: {season}"
    
    def infer_single_image(
        self,
        sar_image_path: str,
        region: str,
        season: str,
        output_path: str = None,
        strength: float = 0.8,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        seed: int = None
    ) -> np.ndarray:
        """
        Perform inference on a single SAR image
        
        Args:
            sar_image_path: Path to input SAR image
            region: Region type (tropical, temperate, arctic)
            season: Season (winter, spring, summer, autumn)
            output_path: Path to save output image (optional)
            strength: Strength of the denoising process (0-1)
            guidance_scale: Classifier-free guidance scale
            num_inference_steps: Number of denoising steps
            seed: Random seed for reproducible results
            
        Returns:
            Generated image as numpy array
        """
        # Load and preprocess SAR image
        sar_image = Image.open(sar_image_path).convert('RGB')
        
        # Generate prompt
        prompt = self.generate_prompt(region, season)
        uncond_prompt = ""  # Empty prompt for classifier-free guidance
        
        print(f"Generating image with prompt: '{prompt}'")
          # Generate image using the pipeline
        generated_image = generate(
            prompt=prompt,
            uncond_prompt=uncond_prompt,
            input_image=sar_image,
            strength=strength,
            do_cfg=True,
            cfg_scale=guidance_scale,  # Note: using cfg_scale parameter name
            sampler_name="ddpm",
            n_inference_steps=num_inference_steps,
            models=self.models,
            seed=seed,
            device=self.device,
            tokenizer=self.tokenizer
        )
          # Convert to numpy array and denormalize
        if isinstance(generated_image, torch.Tensor):
            generated_image = generated_image.cpu().numpy()
        print(f"Generated image shape: {generated_image.shape}")
        print(f"Generated image dtype: {generated_image.dtype}")
        print(f"Generated image min/max: {generated_image.min():.3f}/{generated_image.max():.3f}")
        
        # Handle different possible output shapes
        if len(generated_image.shape) == 4:
            # Remove batch dimension if present: (1, C, H, W) -> (C, H, W)
            generated_image = generated_image.squeeze(0)
        elif len(generated_image.shape) == 2:
            # Handle 2D output: (H, W) -> (H, W, 1) -> (H, W, 3)
            generated_image = np.expand_dims(generated_image, axis=-1)
            generated_image = np.repeat(generated_image, 3, axis=-1)
            # Convert to CHW format for consistency
            generated_image = generated_image.transpose(2, 0, 1)
        elif len(generated_image.shape) == 3:
            # Check if it's in HWC format (height, width, channels)
            if generated_image.shape[2] in [1, 3, 4] and generated_image.shape[2] < generated_image.shape[0]:
                print(f"Detected HWC format: {generated_image.shape}")
                # Convert from HWC to CHW
                generated_image = generated_image.transpose(2, 0, 1)
                print(f"Converted to CHW: {generated_image.shape}")
            # If it's already in CHW format, keep as is
          # Ensure we have the correct shape: (C, H, W)
        if len(generated_image.shape) != 3:
            raise ValueError(f"Unexpected generated image shape: {generated_image.shape}. Expected (C, H, W)")
        
        # Handle unusual channel configurations and shapes
        channels, height, width = generated_image.shape
        
        # If we have very unusual dimensions, try to reshape to a square image
        if height == 1 or width == 1 or height * width < 64 * 64:
            print(f"Warning: Unusual image dimensions {height}x{width}, attempting to reshape...")
            
            # Calculate total pixels across all channels
            total_pixels = channels * height * width
            
            # Try to create a reasonable square image
            if total_pixels >= 256 * 256:
                side_length = int(np.sqrt(total_pixels // 3))
                remaining_pixels = total_pixels - (3 * side_length * side_length)
                
                # If we can make a good square image with 3 channels
                if remaining_pixels < side_length:  # Small remainder is acceptable
                    flat_data = generated_image.flatten()[:3 * side_length * side_length]
                    generated_image = flat_data.reshape(3, side_length, side_length)
                    print(f"Reshaped to: {generated_image.shape}")
                else:
                    # Fall back to 256x256 if we have enough pixels
                    flat_data = generated_image.flatten()
                    if len(flat_data) >= 3 * 256 * 256:
                        generated_image = flat_data[:3 * 256 * 256].reshape(3, 256, 256)
                        print(f"Reshaped to default 256x256: {generated_image.shape}")
                    else:
                        print(f"Insufficient pixels for valid image: {total_pixels}")
                        raise ValueError(f"Cannot create valid image from {generated_image.shape}")
            else:
                print(f"Insufficient pixels for valid image: {total_pixels}")
                raise ValueError(f"Cannot create valid image from {generated_image.shape}")
        
        # Ensure correct number of channels
        if generated_image.shape[0] == 1:
            # Grayscale: (1, H, W) -> (3, H, W)
            generated_image = np.repeat(generated_image, 3, axis=0)
        elif generated_image.shape[0] == 4:
            # 4-channel (RGBA or latent): take first 3 channels
            generated_image = generated_image[:3]
        elif generated_image.shape[0] != 3:
            print(f"Warning: Unusual number of channels {generated_image.shape[0]}, attempting to fix...")
            if generated_image.shape[0] > 3:
                # Take first 3 channels
                generated_image = generated_image[:3]
            else:
                # Repeat channels to get 3
                channels_needed = 3 // generated_image.shape[0]
                remainder = 3 % generated_image.shape[0]
                repeated = np.repeat(generated_image, channels_needed, axis=0)
                if remainder > 0:
                    extra = generated_image[:remainder]
                    generated_image = np.concatenate([repeated, extra], axis=0)
                else:
                    generated_image = repeated
          # Convert from [-1, 1] to [0, 255]
        generated_image = ((generated_image + 1) * 127.5).clip(0, 255).astype(np.uint8)
        
        # Resize from 512x512 to 256x256 if needed (since pipeline generates 512x512)
        if generated_image.shape[1] == 512 and generated_image.shape[2] == 512:
            print("Resizing from 512x512 to 256x256...")
            # Convert to PIL for resizing
            temp_img = Image.fromarray(generated_image.transpose(1, 2, 0))
            temp_img = temp_img.resize((256, 256), Image.LANCZOS)
            generated_image = np.array(temp_img).transpose(2, 0, 1)
        
        print(f"Final image shape: {generated_image.shape}")
        
        # Save output if path provided
        if output_path:
            # Convert from CHW to HWC for PIL
            output_image = Image.fromarray(generated_image.transpose(1, 2, 0))
            output_image.save(output_path)
            print(f"Saved generated image to {output_path}")
        
        return generated_image
    
    def infer_batch(
        self,
        sar_images_dir: str,
        regions: list,
        seasons: list,
        output_dir: str,
        **inference_kwargs
    ):
        """
        Perform batch inference on multiple SAR images
        
        Args:
            sar_images_dir: Directory containing SAR images
            regions: List of regions for each image
            seasons: List of seasons for each image
            output_dir: Directory to save output images
            **inference_kwargs: Additional arguments for inference
        """
        os.makedirs(output_dir, exist_ok=True)
        
        sar_images = list(Path(sar_images_dir).glob("*.png")) + \
                    list(Path(sar_images_dir).glob("*.jpg")) + \
                    list(Path(sar_images_dir).glob("*.jpeg"))
        
        for i, sar_image_path in enumerate(sar_images):
            region = regions[i % len(regions)]
            season = seasons[i % len(seasons)]
            
            output_path = os.path.join(
                output_dir, 
                f"{sar_image_path.stem}_{region}_{season}_generated.png"
            )
            
            print(f"Processing {sar_image_path.name} ({i+1}/{len(sar_images)})")
            
            try:
                self.infer_single_image(
                    str(sar_image_path),
                    region,
                    season,
                    output_path,
                    **inference_kwargs
                )
            except Exception as e:
                print(f"Error processing {sar_image_path.name}: {e}")
                continue
    
    def infer_from_dataset(
        self,
        dataset_path: str,
        output_dir: str,
        num_samples: int = 10,
        **inference_kwargs
    ):
        """
        Perform inference on samples from the dataset
        
        Args:
            dataset_path: Path to the dataset
            output_dir: Directory to save output images
            num_samples: Number of samples to process
            **inference_kwargs: Additional arguments for inference
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load test dataset
        test_dataset = SARDataset(
            dataset_path=dataset_path,
            train=False,
            train_split=0.8
        )
        
        print(f"Processing {min(num_samples, len(test_dataset))} samples from test dataset")
        
        for i in range(min(num_samples, len(test_dataset))):
            sample = test_dataset[i]
            
            # Extract metadata
            metadata = sample['metadata']
            region = metadata['region']
            season = metadata['season']
            
            # Get SAR image path
            # Note: We need to reconstruct the path from the dataset
            # For now, we'll use the sample data directly
            
            output_path = os.path.join(
                output_dir,
                f"sample_{i:03d}_{region}_{season}_generated.png"
            )
            
            print(f"Processing sample {i+1}/{min(num_samples, len(test_dataset))}")
            
            try:
                # For dataset samples, we need to work with the loaded tensor
                # Convert back to PIL image
                sar_tensor = sample['s1_image']
                sar_array = self.denormalize(sar_tensor).cpu().numpy()
                sar_array = ((sar_array + 1) * 127.5).astype(np.uint8)
                sar_image = Image.fromarray(sar_array.transpose(1, 2, 0))
                
                # Generate prompt
                prompt = self.generate_prompt(region, season)
                uncond_prompt = ""
                
                print(f"Generating with prompt: '{prompt}'")
                  # Generate image
                # Map inference_kwargs to correct parameter names
                generate_kwargs = {}
                if 'guidance_scale' in inference_kwargs:
                    generate_kwargs['cfg_scale'] = inference_kwargs['guidance_scale']
                if 'num_inference_steps' in inference_kwargs:
                    generate_kwargs['n_inference_steps'] = inference_kwargs['num_inference_steps']
                if 'strength' in inference_kwargs:
                    generate_kwargs['strength'] = inference_kwargs['strength']
                if 'seed' in inference_kwargs:
                    generate_kwargs['seed'] = inference_kwargs['seed']
                
                generated_image = generate(
                    prompt=prompt,
                    uncond_prompt=uncond_prompt,
                    input_image=sar_image,
                    models=self.models,
                    device=self.device,
                    tokenizer=self.tokenizer,
                    **generate_kwargs
                )                # Save generated image
                if isinstance(generated_image, torch.Tensor):
                    generated_image = generated_image.cpu().numpy()
                
                print(f"Generated image shape: {generated_image.shape}")
                
                # Handle different possible output shapes
                if len(generated_image.shape) == 4:
                    # Remove batch dimension if present: (1, C, H, W) -> (C, H, W)
                    generated_image = generated_image.squeeze(0)
                elif len(generated_image.shape) == 2:
                    # Handle 2D output: (H, W) -> (H, W, 1) -> (H, W, 3)
                    generated_image = np.expand_dims(generated_image, axis=-1)
                    generated_image = np.repeat(generated_image, 3, axis=-1)
                    # Convert to CHW format for consistency
                    generated_image = generated_image.transpose(2, 0, 1)
                elif len(generated_image.shape) == 3:
                    # Check if it's in HWC format (height, width, channels)
                    if generated_image.shape[2] in [1, 3, 4] and generated_image.shape[2] < generated_image.shape[0]:
                        print(f"Detected HWC format: {generated_image.shape}")
                        # Convert from HWC to CHW
                        generated_image = generated_image.transpose(2, 0, 1)
                        print(f"Converted to CHW: {generated_image.shape}")
                    # If it's already in CHW format, keep as is
                
                # Ensure we have the correct shape: (C, H, W)
                if len(generated_image.shape) != 3:
                    print(f"Warning: Unexpected shape {generated_image.shape}, attempting to reshape...")
                    # Try to reshape to a reasonable image shape
                    total_elements = generated_image.size
                    if total_elements >= 256 * 256:
                        # Try to create a square image
                        side_length = int(np.sqrt(total_elements // 3))
                        generated_image = generated_image.reshape(3, side_length, side_length)
                    else:
                        print(f"Cannot create valid image from shape {generated_image.shape}")
                        continue
                
                # Ensure correct number of channels
                if generated_image.shape[0] == 1:
                    # Grayscale: (1, H, W) -> (3, H, W)
                    generated_image = np.repeat(generated_image, 3, axis=0)
                elif generated_image.shape[0] == 4:
                    # 4-channel: take first 3 channels
                    generated_image = generated_image[:3]
                  # Convert from [-1, 1] to [0, 255]
                generated_image = ((generated_image + 1) * 127.5).clip(0, 255).astype(np.uint8)
                
                # Resize from 512x512 to 256x256 if needed (since pipeline generates 512x512)
                if generated_image.shape[1] == 512 and generated_image.shape[2] == 512:
                    print("Resizing from 512x512 to 256x256...")
                    # Convert to PIL for resizing
                    temp_img = Image.fromarray(generated_image.transpose(1, 2, 0))
                    temp_img = temp_img.resize((256, 256), Image.LANCZOS)
                    generated_image = np.array(temp_img).transpose(2, 0, 1)
                
                # Convert from CHW to HWC for PIL
                output_image = Image.fromarray(generated_image.transpose(1, 2, 0))
                output_image.save(output_path)
                
                print(f"Saved generated image to {output_path}")
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue


def main():
    parser = argparse.ArgumentParser(description="SAR Image Colorization Inference")
    parser.add_argument("--model_checkpoint", type=str, required=True,
                       help="Path to the trained model checkpoint")
    parser.add_argument("--clip_checkpoint", type=str, default=None,
                       help="Path to the trained CLIP checkpoint")
    parser.add_argument("--mode", type=str, choices=["single", "batch", "dataset"], 
                       default="single", help="Inference mode")
    
    # Single image mode arguments
    parser.add_argument("--input_image", type=str, help="Input SAR image path")
    parser.add_argument("--region", type=str, choices=["tropical", "temperate", "arctic"],
                       help="Region type")
    parser.add_argument("--season", type=str, choices=["winter", "spring", "summer", "autumn", "fall"],
                       help="Season")
    parser.add_argument("--output_path", type=str, help="Output image path")
    
    # Batch mode arguments
    parser.add_argument("--input_dir", type=str, help="Input directory for batch mode")
    parser.add_argument("--regions", type=str, nargs="+", 
                       choices=["tropical", "temperate", "arctic"],
                       help="List of regions for batch mode")
    parser.add_argument("--seasons", type=str, nargs="+",
                       choices=["winter", "spring", "summer", "autumn", "fall"],
                       help="List of seasons for batch mode")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    
    # Dataset mode arguments
    parser.add_argument("--dataset_path", type=str, help="Path to dataset")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of samples to process in dataset mode")
    
    # Inference parameters
    parser.add_argument("--strength", type=float, default=0.8,
                       help="Denoising strength (0-1)")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                       help="Classifier-free guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                       help="Number of denoising steps")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducible results")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"], help="Device to use")
    
    args = parser.parse_args()
    
    # Initialize inference pipeline
    inference = SARInference(
        model_checkpoint_path=args.model_checkpoint,
        clip_checkpoint_path=args.clip_checkpoint,
        device=args.device
    )
    
    # Inference parameters
    inference_kwargs = {
        "strength": args.strength,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed
    }
    
    # Run inference based on mode
    if args.mode == "single":
        if not all([args.input_image, args.region, args.season]):
            raise ValueError("Single mode requires --input_image, --region, and --season")
        
        inference.infer_single_image(
            sar_image_path=args.input_image,
            region=args.region,
            season=args.season,
            output_path=args.output_path or f"generated_{args.region}_{args.season}.png",
            **inference_kwargs
        )
    
    elif args.mode == "batch":
        if not all([args.input_dir, args.regions, args.seasons, args.output_dir]):
            raise ValueError("Batch mode requires --input_dir, --regions, --seasons, and --output_dir")
        
        inference.infer_batch(
            sar_images_dir=args.input_dir,
            regions=args.regions,
            seasons=args.seasons,
            output_dir=args.output_dir,
            **inference_kwargs
        )
    
    elif args.mode == "dataset":
        if not all([args.dataset_path, args.output_dir]):
            raise ValueError("Dataset mode requires --dataset_path and --output_dir")
        
        inference.infer_from_dataset(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            **inference_kwargs
        )


if __name__ == "__main__":
    main()
