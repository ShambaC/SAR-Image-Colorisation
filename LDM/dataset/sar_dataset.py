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
    Supports proper train/validation/test splits with reproducible data splitting.
    """
    
    def __init__(self, split, im_path, im_size=256, im_channels=3, im_ext='png',
                 use_latents=False, latent_path=None, condition_config=None,
                 metadata_file=None, train_split=0.7, val_split=0.15, test_split=0.15, 
                 random_seed=42):
        """
        Initialize SAR Dataset with proper train/val/test splitting
        
        Args:
            split: 'train', 'val', or 'test'
            im_path: Path to dataset directory
            im_size: Image size for resizing
            im_channels: Number of image channels
            im_ext: Image file extension
            use_latents: Whether to use precomputed latents
            latent_path: Path to latent files
            condition_config: Configuration for conditioning types
            metadata_file: Specific metadata CSV file to use (if None, loads from split-specific files)
            train_split: Proportion for training set
            val_split: Proportion for validation set  
            test_split: Proportion for test set
            random_seed: Random seed for reproducible splits
        """
        assert split in ['train', 'val', 'test'], f"Split must be 'train', 'val', or 'test', got {split}"
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Split proportions must sum to 1.0"
        
        self.split = split
        self.im_size = im_size
        self.im_channels = im_channels
        self.im_ext = im_ext
        self.im_path = im_path
        self.latent_maps = None
        self.use_latents = True
        self.random_seed = random_seed
        
        # Get condition types
        self.condition_types = [] if condition_config is None else condition_config['condition_types']
        
        # Load dataset with proper splitting
        self.s1_images, self.s2_images, self.text_prompts = self.load_dataset_with_splits(
            im_path, metadata_file, train_split, val_split, test_split)
          # Whether to load images or to load latents
        if use_latents and latent_path is not None:
            latent_maps = load_latents(latent_path)
            if len(latent_maps) == len(self.s2_images):
                self.use_latents = True
                self.latent_maps = latent_maps
                print('Found {} latents'.format(len(self.latent_maps)))
            else:
                print('Latents not found')
    
    def load_dataset_with_splits(self, dataset_path, metadata_file, train_split, val_split, test_split):
        """
        Load SAR-optical image pairs with proper train/val/test splitting
        
        Two modes:
        1. If metadata_file is provided: Use that specific CSV file and split it
        2. If metadata_file is None: Look for pre-split files (train_metadata.csv, val_metadata.csv, test_metadata.csv)
           or load all CSV files and create splits
        """
        assert os.path.exists(dataset_path), f"Dataset path {dataset_path} does not exist"
        
        s1_images = []
        s2_images = []
        text_prompts = []
        
        # Check for pre-split metadata files
        pre_split_files = {
            'train': os.path.join(dataset_path, 'train_metadata.csv'),
            'val': os.path.join(dataset_path, 'val_metadata.csv'), 
            'test': os.path.join(dataset_path, 'test_metadata.csv')
        }
        
        if metadata_file is not None:
            # Use specific metadata file and split it
            print(f"Using specific metadata file: {metadata_file}")
            df_all = pd.read_csv(metadata_file)
            df_split = self._create_splits(df_all, train_split, val_split, test_split)
            
        elif all(os.path.exists(f) for f in pre_split_files.values()):
            # Use pre-existing split files
            print(f"Using pre-split metadata files for {self.split}")
            df_split = pd.read_csv(pre_split_files[self.split])
            
        else:
            # Load all CSV files and create splits
            print("Creating splits from all available CSV files")
            csv_files = glob.glob(os.path.join(dataset_path, '*_metadata.csv'))
            if not csv_files:
                # Fallback to legacy naming
                csv_files = glob.glob(os.path.join(dataset_path, 'data_r_*.csv'))
            
            if not csv_files:
                raise FileNotFoundError(f"No metadata CSV files found in {dataset_path}")
            
            print(f"Found {len(csv_files)} CSV files")
            
            # Combine all CSV files
            all_dfs = []
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                # Add source file info for tracking
                df['source_file'] = os.path.basename(csv_file)
                all_dfs.append(df)
            
            df_all = pd.concat(all_dfs, ignore_index=True)
            df_split = self._create_splits(df_all, train_split, val_split, test_split)
        
        # Process the data for current split
        print(f"Processing {len(df_split)} samples for {self.split} split")
        
        for _, row in tqdm(df_split.iterrows(), total=len(df_split), desc=f"Loading {self.split} data"):
            # Handle different possible column names
            if 'region_folder' in row and 's1_folder' in row and 's2_folder' in row and 'image_name' in row:
                # New format with explicit folder structure
                s1_path = os.path.join(dataset_path, row['region_folder'], row['s1_folder'], row['image_name'])
                s2_path = os.path.join(dataset_path, row['region_folder'], row['s2_folder'], row['image_name'])
            elif 's1_fileName' in row and 's2_fileName' in row:
                # Legacy format with full file paths
                s1_path = os.path.join(dataset_path, row['s1_fileName'].replace("\\", "/"))
                s2_path = os.path.join(dataset_path, row['s2_fileName'].replace("\\", "/"))
            else:
                print(f"Warning: Unknown CSV format, skipping row {row}")
                continue
            
            # Check if both images exist
            if os.path.exists(s1_path) and os.path.exists(s2_path):
                s1_images.append(s1_path)
                s2_images.append(s2_path)
                
                # Generate text prompt from metadata
                region = row.get('region', 'unknown')
                season = row.get('season', 'unknown')
                if pd.isna(region):
                    region = 'unknown'
                if pd.isna(season):
                    season = 'unknown'
                    
                text_prompt = f"Colorise image, Region: {region}, Season: {season}"
                text_prompts.append(text_prompt)
            else:
                print(f"Warning: Missing images - S1: {os.path.exists(s1_path)}, S2: {os.path.exists(s2_path)}")
        
        print(f'Loaded {len(s1_images)} SAR images for {self.split}')
        print(f'Loaded {len(s2_images)} optical images for {self.split}')
        print(f'Generated {len(text_prompts)} text prompts for {self.split}')
        
        assert len(s1_images) == len(s2_images) == len(text_prompts), \
            "Mismatch in number of SAR images, optical images, and text prompts"
        
        return s1_images, s2_images, text_prompts
    
    def _create_splits(self, df_all, train_split, val_split, test_split):
        """
        Create train/val/test splits from combined dataframe
        Uses stratified splitting based on region if available
        """
        # Set random seed for reproducible splits
        np.random.seed(self.random_seed)
        
        # Shuffle the data
        df_shuffled = df_all.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        n_total = len(df_shuffled)
        n_train = int(train_split * n_total)
        n_val = int(val_split * n_total)
        
        # Create splits
        if self.split == 'train':
            df_split = df_shuffled[:n_train]
        elif self.split == 'val':
            df_split = df_shuffled[n_train:n_train + n_val]
        else:  # test
            df_split = df_shuffled[n_train + n_val:]
        
        return df_split
    
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
