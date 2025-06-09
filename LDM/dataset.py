import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from typing import Tuple, List, Dict, Optional
import glob
import random


def custom_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function to handle mixed data types in batches.
    
    Args:
        batch: List of data samples from the dataset
        
    Returns:
        Dictionary with batched tensors and lists of strings
    """
    # Stack image tensors
    s1_images = torch.stack([item['s1_image'] for item in batch])
    s2_images = torch.stack([item['s2_image'] for item in batch])
    
    # Keep text prompts as list of strings
    prompts = [item['prompt'] for item in batch]
    
    # Keep metadata as list of dictionaries
    metadata = [item['metadata'] for item in batch]
    
    return {
        's1_image': s1_images,
        's2_image': s2_images,
        'prompt': prompts,
        'metadata': metadata
    }


class SARDataset(Dataset):
    """Dataset for SAR to Optical Image Translation with text prompts"""
    
    def __init__(
        self, 
        dataset_path: str, 
        train: bool = True,
        train_split: float = 0.8,
        transform: Optional[transforms.Compose] = None,
        seed: int = 42
    ):
        """
        Args:
            dataset_path: Path to the Dataset folder
            train: Whether to load train or test split
            train_split: Fraction of data to use for training
            transform: Optional transforms to apply to images
            seed: Random seed for reproducible splits
        """
        self.dataset_path = dataset_path
        self.train = train
        self.transform = transform or self.get_default_transforms()
        
        # Load all CSV files and create dataset
        self.data = self._load_dataset(train_split, seed)
        
    def _load_dataset(self, train_split: float, seed: int) -> List[Dict]:
        """Load and split dataset from CSV files"""
        # Find all CSV files in the dataset
        csv_files = glob.glob(os.path.join(self.dataset_path, "data_r_*.csv"))
        
        all_data = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                # Fix path separators for unix compatibility
                s1_path = row['s1_fileName'].replace('\\', '/')
                s2_path = row['s2_fileName'].replace('\\', '/')
                
                # Create text prompt
                prompt = f"Colorise image, Region: {row['region']}, Season: {row['season']}"
                
                data_point = {
                    's1_path': os.path.join(self.dataset_path, s1_path),
                    's2_path': os.path.join(self.dataset_path, s2_path),
                    'prompt': prompt,
                    'region': row['region'],
                    'season': row['season'],
                    'country': row['country'],
                    'coordinates': row['coordinates'],
                    'date_time': row['date-time'],
                    'scale': row['scale'],
                    'operational_mode': row['operational-mode'],
                    'polarisation': row['polarisation'],
                    'bands': row['bands']
                }
                
                # Only add if both images exist
                if os.path.exists(data_point['s1_path']) and os.path.exists(data_point['s2_path']):
                    all_data.append(data_point)
        
        # Shuffle with seed for reproducible splits
        random.seed(seed)
        random.shuffle(all_data)
        
        # Split data
        split_idx = int(len(all_data) * train_split)
        if self.train:
            return all_data[:split_idx]
        else:
            return all_data[split_idx:]
    
    def get_default_transforms(self) -> transforms.Compose:
        """Default transforms for 256x256 images"""
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary containing:
            - 's1_image': SAR image tensor (C, H, W)
            - 's2_image': Optical image tensor (C, H, W)  
            - 'prompt': Text prompt string
            - 'metadata': Additional metadata
        """
        data_point = self.data[idx]
        
        # Load images
        try:
            s1_image = Image.open(data_point['s1_path']).convert('RGB')
            s2_image = Image.open(data_point['s2_path']).convert('RGB')
        except Exception as e:
            print(f"Error loading images at idx {idx}: {e}")
            # Return a random valid sample instead
            return self.__getitem__(random.randint(0, len(self.data) - 1))
        
        # Apply transforms
        s1_image = self.transform(s1_image)
        s2_image = self.transform(s2_image)
        
        return {
            's1_image': s1_image,
            's2_image': s2_image,
            'prompt': data_point['prompt'],
            'metadata': {
                'region': data_point['region'],
                'season': data_point['season'],
                'country': data_point['country'],
                'coordinates': data_point['coordinates'],
                'date_time': data_point['date_time'],
                'scale': data_point['scale'],
                'operational_mode': data_point['operational_mode'],
                'polarisation': data_point['polarisation'],
                'bands': data_point['bands']
            }
        }


def create_dataloaders(
    dataset_path: str,
    batch_size: int = 4,
    train_split: float = 0.8,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders
    
    Args:
        dataset_path: Path to the Dataset folder
        batch_size: Batch size for training
        train_split: Fraction of data for training
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        seed: Random seed
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    
    train_dataset = SARDataset(
        dataset_path=dataset_path,
        train=True,
        train_split=train_split,
        seed=seed
    )
    
    test_dataset = SARDataset(
        dataset_path=dataset_path,
        train=False,
        train_split=train_split,
        seed=seed
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=custom_collate_fn  # Use custom collate function
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=custom_collate_fn  # Use custom collate function
    )
    
    return train_loader, test_loader


def get_dataset_statistics(dataset_path: str) -> Dict:
    """Get statistics about the dataset"""
    csv_files = glob.glob(os.path.join(dataset_path, "data_r_*.csv"))
    
    total_samples = 0
    regions = set()
    seasons = set()
    countries = set()
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            total_samples += len(df)
            
            # Convert to string and handle NaN values
            region_values = df['region'].astype(str).replace('nan', '').unique()
            season_values = df['season'].astype(str).replace('nan', '').unique()
            country_values = df['country'].astype(str).replace('nan', '').unique()
            
            # Filter out empty strings
            regions.update([r for r in region_values if r and r != ''])
            seasons.update([s for s in season_values if s and s != ''])
            countries.update([c for c in country_values if c and c != ''])
        except Exception as e:
            print(f"Warning: Error reading {csv_file}: {e}")
            continue
    
    return {
        'total_samples': total_samples,
        'regions': sorted(list(regions)),
        'seasons': sorted(list(seasons)),
        'countries': sorted(list(countries)),
        'num_csv_files': len(csv_files)
    }


if __name__ == "__main__":
    # Test the dataset
    dataset_path = "../Dataset"
    
    print("Dataset Statistics:")
    stats = get_dataset_statistics(dataset_path)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\nTesting dataset loading...")
    train_loader, test_loader = create_dataloaders(
        dataset_path=dataset_path,
        batch_size=2,
        num_workers=0  # Set to 0 for testing
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test loading a batch
    for batch in train_loader:
        print(f"S1 image shape: {batch['s1_image'].shape}")
        print(f"S2 image shape: {batch['s2_image'].shape}")
        print(f"Sample prompts: {batch['prompt'][:2]}")
        break
