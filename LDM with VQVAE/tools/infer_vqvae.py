import argparse
import glob
import os
import pickle

import torch
import torchvision
import yaml
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from dataset.sar_dataset import SARDataset

from models.vqvae import VQVAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def infer(args):
    ######## Read the config file #######
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    
    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']
    
    # Create the dataset
    im_dataset_cls = {
        'sar': SARDataset,
    }.get(dataset_config['name'])
    
    im_dataset = im_dataset_cls(split='train',
                                im_path=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'],
                                return_path=True)  # Enable path return for latent saving
    
    # This is only used for saving latents. Which as of now
    # is not done in batches hence batch size 1
    data_loader = DataLoader(im_dataset,
                             batch_size=1,
                             shuffle=False)

    num_images = train_config['num_samples']
    ngrid = train_config['num_grid_rows']
    
    idxs = torch.randint(0, len(im_dataset) - 1, (num_images,))
    # Handle the fact that dataset now returns tuples when return_path=True
    ims = []
    for idx in idxs:
        data = im_dataset[idx]
        if isinstance(data, (list, tuple)):
            im = data[0]  # First element is always the image tensor
        else:
            im = data
        ims.append(im[None, :])
    ims = torch.cat(ims).float()
    ims = ims.to(device)
    
    model = VQVAE(im_channels=dataset_config['im_channels'],
                  model_config=autoencoder_config).to(device)
    model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                    train_config['vqvae_autoencoder_ckpt_name']),
                                     map_location=device))
    model.eval()
    
    with torch.no_grad():
        
        encoded_output, _ = model.encode(ims)
        decoded_output = model.decode(encoded_output)
        encoded_output = torch.clamp(encoded_output, -1., 1.)
        encoded_output = (encoded_output + 1) / 2
        decoded_output = torch.clamp(decoded_output, -1., 1.)
        decoded_output = (decoded_output + 1) / 2
        ims = (ims + 1) / 2

        encoder_grid = make_grid(encoded_output.cpu(), nrow=ngrid)
        decoder_grid = make_grid(decoded_output.cpu(), nrow=ngrid)
        input_grid = make_grid(ims.cpu(), nrow=ngrid)
        encoder_grid = torchvision.transforms.ToPILImage()(encoder_grid)
        decoder_grid = torchvision.transforms.ToPILImage()(decoder_grid)
        input_grid = torchvision.transforms.ToPILImage()(input_grid)
        
        input_grid.save(os.path.join(train_config['task_name'], 'input_samples.png'))
        encoder_grid.save(os.path.join(train_config['task_name'], 'encoded_samples.png'))
        decoder_grid.save(os.path.join(train_config['task_name'], 'reconstructed_samples.png'))
        
        if train_config['save_latents']:
            # save Latents (but in a very unoptimized way)
            latent_path = os.path.join(train_config['task_name'], train_config['vqvae_latent_dir_name'])
            latent_fnames = glob.glob(os.path.join(train_config['task_name'], train_config['vqvae_latent_dir_name'],
                                                   '*.pt'))
            assert len(latent_fnames) == 0, 'Latents already present. Delete all latent files and re-run'
            if not os.path.exists(latent_path):
                os.mkdir(latent_path)
            print('Saving Latents for {}'.format(dataset_config['name']))
            
            for idx, data in enumerate(tqdm(data_loader)):
                if isinstance(data, (list, tuple)):
                    # When return_path=True, the last element is the file path
                    if len(data) == 2:  # (image, path)
                        im, color_image_path = data
                    elif len(data) == 3:  # (image, conditions, path)
                        im, _, color_image_path = data
                    else:
                        im = data[0]
                        color_image_path = f"image_{idx}"  # fallback
                else:
                    im = data
                    color_image_path = f"image_{idx}"  # fallback
                
                encoded_output, _ = model.encode(im.float().to(device))
                
                # Create a safe filename from the path
                safe_filename = color_image_path[0].replace('/', '_').replace('\\', '_').replace(':', '_')
                latent_filename = f"{safe_filename}.pt"
                torch.save(encoded_output.cpu(), os.path.join(latent_path, latent_filename))
            print('Done saving latents')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vq vae inference')
    parser.add_argument('--config', dest='config_path',
                        default='config/sar_colorization.yaml', type=str)
    args = parser.parse_args()
    infer(args)
