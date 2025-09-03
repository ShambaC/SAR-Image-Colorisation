import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class SARColorizationDataset(Dataset):
    def __init__(self, im_path, im_size, im_channels, split='train', use_latents=False, latent_path=None, condition_config=None, **kwargs):
        self.im_path = im_path
        self.im_size = im_size
        self.im_channels = im_channels
        self.use_latents = use_latents
        self.latent_path = latent_path
        self.condition_config = condition_config
        self.split = split

        csv_files = glob.glob(os.path.join(self.im_path, 'data_r_*.csv'))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.im_path}")
            
        df_list = [pd.read_csv(f) for f in csv_files]
        self.metadata = pd.concat(df_list, ignore_index=True).dropna()

        self.transform = transforms.Compose([
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.gray_transform = transforms.Compose([
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata.iloc[idx]
        
        base_data_path = os.path.dirname(self.im_path)

        s1_path = os.path.join(base_data_path, item['s1_fileName'].replace('\\', '/'))
        s2_path = os.path.join(base_data_path, item['s2_fileName'].replace('\\', '/'))
        
        s2_image = Image.open(s2_path).convert('RGB')

        if self.condition_config is None:
            return self.transform(s2_image)

        if self.use_latents:
            s2_fname_without_ext = os.path.splitext(os.path.basename(item['s2_fileName']))[0]
            latent_filename = f"{s2_fname_without_ext}.pt"
            latent_full_path = os.path.join(self.latent_path, latent_filename)
            main_data = torch.load(latent_full_path)
        else:
            main_data = self.transform(s2_image)

        cond_input = {}
        condition_types = self.condition_config.get('condition_types', [])

        if 'text' in condition_types:
            region = item['region']
            season = item['season']
            text_prompt = f"Colorize the image, Region: {region}, Season: {season}"
            cond_input['text'] = text_prompt
        
        if 'image' in condition_types:
            s1_image = Image.open(s1_path).convert('L')
            cond_input['image'] = self.gray_transform(s1_image)
            
        return main_data, cond_input
