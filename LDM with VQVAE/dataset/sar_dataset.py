import glob
import os
import random
import torch
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
from utils.diffusion_utils import load_latents
from tqdm import tqdm
from torch.utils.data.dataset import Dataset


class SARDataset(Dataset):
    r"""
    SAR dataset for SAR image colorization.
    This dataset loads grayscale SAR images from Sentinel-1 and 
    corresponding color images from Sentinel-2, along with region and season
    information to generate text prompts for conditioning.
    """
    
    def __init__(self, split, im_path, im_size=256, im_channels=3, im_ext='jpg',
                 use_latents=False, latent_path=None, condition_config=None):
        self.split = split
        self.im_size = im_size
        self.im_channels = im_channels
        self.im_ext = im_ext
        self.im_path = im_path
        self.latent_maps = None
        self.use_latents = False
        
        self.condition_types = [] if condition_config is None else condition_config['condition_types']
        
        self.idx_to_cls_map = {}
        self.cls_to_idx_map = {}
        
        self.sar_images, self.color_images, self.texts = self.load_images(im_path)
        
        # Whether to load images or to load latents
        if use_latents and latent_path is not None:
            latent_maps = load_latents(latent_path)
            if len(latent_maps) == len(self.color_images):
                self.use_latents = True
                self.latent_maps = latent_maps
                print('Found {} latents'.format(len(self.latent_maps)))
            else:
                print('Latents not found')
    
    def load_images(self, im_path):
        r"""
        Loads SAR and color image pairs from the SAR_Dataset directory structure
        and generates text prompts from region and season information.
        """
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)
        
        sar_images = []
        color_images = []
        texts = []
        
        # Look for SAR_Dataset folder
        sar_dataset_path = os.path.join(im_path, 'SAR_Dataset')
        assert os.path.exists(sar_dataset_path), "SAR_Dataset folder not found in {}".format(im_path)
        
        # Find all r_XXX folders
        r_folders = glob.glob(os.path.join(sar_dataset_path, 'r_*'))
        r_folders = [f for f in r_folders if os.path.isdir(f)]
        
        print(f"Found {len(r_folders)} region folders")
        
        for r_folder in tqdm(r_folders, desc="Loading SAR dataset"):
            # Extract region number from folder name
            r_name = os.path.basename(r_folder)
            
            # Look for corresponding CSV file
            csv_file = os.path.join(sar_dataset_path, f'data_{r_name}.csv')
            if not os.path.exists(csv_file):
                print(f"Warning: CSV file {csv_file} not found, skipping {r_name}")
                continue
            
            # Read CSV file
            try:
                df = pd.read_csv(csv_file)
                required_columns = ['s1_fileName', 's2_fileName', 'region', 'season']
                if not all(col in df.columns for col in required_columns):
                    print(f"Warning: CSV file {csv_file} missing required columns, skipping")
                    continue
            except Exception as e:
                print(f"Warning: Error reading CSV file {csv_file}: {e}, skipping")
                continue
            
            # Process each row in the CSV
            for _, row in df.iterrows():
                try:
                    # Get file paths (fix double backslashes to forward slashes)
                    s1_file = str(row['s1_fileName']).replace('\\\\', '/').replace('\\', '/')
                    s2_file = str(row['s2_fileName']).replace('\\\\', '/').replace('\\', '/')
                    
                    # Convert to absolute paths relative to the data folder
                    s1_path = os.path.join(im_path, s1_file)
                    s2_path = os.path.join(im_path, s2_file)
                    
                    # Check if files exist
                    if not os.path.exists(s1_path):
                        print(f"Warning: SAR image not found: {s1_path}")
                        continue
                    if not os.path.exists(s2_path):
                        print(f"Warning: Color image not found: {s2_path}")
                        continue
                    
                    # Get region and season
                    region = str(row['region']).lower()
                    season = str(row['season']).lower()
                    
                    # Validate region and season values
                    valid_regions = ['tropical', 'temperate', 'arctic']
                    valid_seasons = ['winter', 'summer', 'spring', 'fall']
                    
                    if region not in valid_regions:
                        print(f"Warning: Invalid region '{region}' in {csv_file}, skipping")
                        continue
                    if season not in valid_seasons:
                        print(f"Warning: Invalid season '{season}' in {csv_file}, skipping")
                        continue
                    
                    # Add to lists
                    sar_images.append(s1_path)
                    color_images.append(s2_path)
                    
                    # Generate text prompt only if text conditioning is enabled
                    if 'text' in self.condition_types:
                        text_prompt = f"Colorize the image, Region: {region}, Season: {season}"
                        texts.append(text_prompt)
                    
                except Exception as e:
                    print(f"Warning: Error processing row in {csv_file}: {e}, skipping")
                    continue
        
        if 'text' in self.condition_types:
            assert len(texts) == len(color_images), "Condition Type Text but could not find text prompts for all images"
        
        print('Found {} SAR images'.format(len(sar_images)))
        print('Found {} color images'.format(len(color_images)))
        print('Found {} text prompts'.format(len(texts)))
        
        return sar_images, color_images, texts
    
    def __len__(self):
        return len(self.color_images)
    
    def __getitem__(self, index):
        ######## Set Conditioning Info ########
        cond_inputs = {}
        if 'text' in self.condition_types:
            cond_inputs['text'] = self.texts[index]
        #######################################
        
        if self.use_latents:
            latent = self.latent_maps[self.color_images[index]]
            if len(self.condition_types) == 0:
                return latent
            else:
                return latent, cond_inputs
        else:
            # Load color image (target)
            color_im = Image.open(self.color_images[index])
            if color_im.mode != 'RGB':
                color_im = color_im.convert('RGB')
            
            color_tensor = torchvision.transforms.Compose([
                torchvision.transforms.Resize(self.im_size),
                torchvision.transforms.CenterCrop(self.im_size),
                torchvision.transforms.ToTensor(),
            ])(color_im)
            color_im.close()
        
            # Convert input to -1 to 1 range.
            color_tensor = (2 * color_tensor) - 1
            
            if len(self.condition_types) == 0:
                return color_tensor
            else:
                return color_tensor, cond_inputs