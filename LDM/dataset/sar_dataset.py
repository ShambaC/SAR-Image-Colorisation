import glob
import os
import pandas as pd
import torch
import torchvision
import numpy as np
from PIL import Image
from utils.diffusion_utils import load_latents
from tqdm import tqdm
from torch.utils.data.dataset import Dataset


class SARDataset(Dataset):
    """
    SAR Dataset for image-to-image translation with text conditioning.
    Loads paired SAR (Sentinel-1) and optical (Sentinel-2) images with metadata.
    """
    
    def __init__(self, split, im_path, im_size=256, im_channels=3, im_ext='png',
                 use_latents=False, latent_path=None, condition_config=None):
        self.split = split
        self.im_size = im_size
        self.im_channels = im_channels
        self.im_ext = im_ext
        self.im_path = im_path
        self.latent_maps = None
        self.use_latents = False
        
        # Get condition types
        self.condition_types = [] if condition_config is None else condition_config['condition_types']
        
        # Load dataset
        self.s1_images, self.s2_images, self.text_prompts = self.load_dataset(im_path)
        
        # Whether to load images or to load latents
        if use_latents and latent_path is not None:
            latent_maps = load_latents(latent_path)
            if len(latent_maps) == len(self.s2_images):
                self.use_latents = True
                self.latent_maps = latent_maps
                print('Found {} latents'.format(len(self.latent_maps)))
            else:
                print('Latents not found')
    
    def load_dataset(self, dataset_path):
        """
        Load all SAR-optical image pairs with metadata from CSV files
        """
        assert os.path.exists(dataset_path), "Dataset path {} does not exist".format(dataset_path)
        
        s1_images = []
        s2_images = []
        text_prompts = []
        
        # Find all CSV files
        csv_files = glob.glob(os.path.join(dataset_path, 'data_r_*.csv'))
        
        print(f"Found {len(csv_files)} CSV files")
        
        for csv_file in tqdm(csv_files, desc="Loading dataset"):
            # Read metadata CSV
            df = pd.read_csv(csv_file)
            
            for _, row in df.iterrows():
                s1_path = os.path.join(dataset_path, row['s1_fileName'])
                s2_path = os.path.join(dataset_path, row['s2_fileName'])
                
                # Check if both images exist
                if os.path.exists(s1_path) and os.path.exists(s2_path):
                    s1_images.append(s1_path)
                    s2_images.append(s2_path)
                    
                    # Generate text prompt from metadata
                    region = row['region'] if pd.notna(row['region']) else 'unknown'
                    season = row['season'] if pd.notna(row['season']) else 'unknown'
                    text_prompt = f"Colorise image, Region: {region}, Season: {season}"
                    text_prompts.append(text_prompt)
        
        print(f'Found {len(s1_images)} SAR images')
        print(f'Found {len(s2_images)} optical images')
        print(f'Generated {len(text_prompts)} text prompts')
        
        assert len(s1_images) == len(s2_images) == len(text_prompts), \
            "Mismatch in number of SAR images, optical images, and text prompts"
            
        return s1_images, s2_images, text_prompts
    
    def load_image(self, image_path):
        """Load and preprocess an image"""
        im = Image.open(image_path)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        
        im_tensor = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.im_size),
            torchvision.transforms.CenterCrop(self.im_size),
            torchvision.transforms.ToTensor(),
        ])(im)
        im.close()
        
        # Convert input to -1 to 1 range
        im_tensor = (2 * im_tensor) - 1
        return im_tensor
    
    def __len__(self):
        return len(self.s2_images)
    
    def __getitem__(self, index):
        """
        Returns target optical image and conditioning inputs (SAR image + text prompt)
        """
        ######## Set Conditioning Info ########
        cond_inputs = {}
        
        # Add text conditioning
        if 'text' in self.condition_types:
            cond_inputs['text'] = self.text_prompts[index]
        
        # Add SAR image conditioning
        if 'image' in self.condition_types:
            s1_tensor = self.load_image(self.s1_images[index])
            cond_inputs['image'] = s1_tensor
        #######################################
        
        # Load target optical image
        if self.use_latents:
            # Use precomputed latents for faster training
            latent = self.latent_maps[self.s2_images[index]]
            if len(self.condition_types) == 0:
                return latent
            else:
                return latent, cond_inputs
        else:
            # Load and process optical image directly
            s2_tensor = self.load_image(self.s2_images[index])
            
            if len(self.condition_types) == 0:
                return s2_tensor
            else:
                return s2_tensor, cond_inputs
