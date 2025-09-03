import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import glob

class SARTrainDataset(Dataset):
    """
    Dataset for training the Diffusion model.
    Reads pairs of Sentinel-1 (grayscale) and Sentinel-2 (color) images
    and generates text prompts from associated metadata.
    """
    def __init__(self, data_root, image_size=(512, 512)):
        self.data_root = data_root
        self.image_size = image_size
        
        # Find all data CSV files and concatenate them
        csv_files = glob.glob(os.path.join(data_root, 'data_r_*.csv'))
        df_list = [pd.read_csv(f) for f in csv_files]
        self.metadata = pd.concat(df_list, ignore_index=True)

        self.s1_files = self.metadata['s1_fileName']
        self.s2_files = self.metadata['s2_fileName']
        self.regions = self.metadata['region']
        self.seasons = self.metadata['season']

        self.transform = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]), # Normalize to [-1, 1]
        ])
        
        # Specific transform for grayscale to ensure it has 3 channels for VAE encoder
        self.s1_transform = T.Compose([
            T.Resize(self.image_size),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # Normalize to [-1, 1]
        ])


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Load images
        s1_path = os.path.join(self.data_root, self.s1_files[idx])
        s2_path = os.path.join(self.data_root, self.s2_files[idx])
        
        s1_img = Image.open(s1_path).convert("RGB") # Convert to RGB
        s2_img = Image.open(s2_path).convert("RGB")

        # Apply transformations
        s1_tensor = self.s1_transform(s1_img)
        s2_tensor = self.transform(s2_img)
        
        # Generate text prompt
        region = self.regions[idx]
        season = self.seasons[idx]
        prompt = f"Colorize the image, Region: {region}, Season: {season}"
        
        return {
            's1_image': s1_tensor,  # Grayscale condition image
            's2_image': s2_tensor,  # Color target image
            'prompt': prompt
        }

class VAEPretrainDataset(Dataset):
    """
    Dataset for pre-training the VAE.
    Only uses the Sentinel-2 color images for reconstruction.
    """
    def __init__(self, data_root, image_size=(512, 512)):
        self.data_root = data_root
        self.image_size = image_size
        
        # Find all Sentinel-2 image files
        r_folders = glob.glob(os.path.join(data_root, 'r_*'))
        self.image_paths = []
        for r_folder in r_folders:
            s2_folder = glob.glob(os.path.join(r_folder, 's2_*'))
            if s2_folder:
                self.image_paths.extend(glob.glob(os.path.join(s2_folder[0], '*')))

        self.transform = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # Normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image)