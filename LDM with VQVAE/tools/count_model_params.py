import argparse
import yaml
import torch
import os
from models.unet_cond_base import Unet


def human(n):
    if n > 1e9:
        return f"{n/1e9:.3f}B"
    if n > 1e6:
        return f"{n/1e6:.3f}M"
    if n > 1e3:
        return f"{n/1e3:.3f}K"
    return str(n)


def main():
    parser = argparse.ArgumentParser(description='Count model parameters for LDM Unet')
    parser.add_argument('--config', '-c', default='config/sar_colorization.yaml', help='Path to YAML config')
    parser.add_argument('--ckpt', default=None, help='Checkpoint path to load (optional)')
    parser.add_argument('--device', default='cuda', help='Device to map the model to (cpu or cuda)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    autoencoder_cfg = cfg.get('autoencoder_params', {})
    ldm_cfg = cfg.get('ldm_params', {})
    train_cfg = cfg.get('train_params', {})

    z_channels = autoencoder_cfg.get('z_channels')
    if z_channels is None:
        raise RuntimeError('z_channels not set in autoencoder_params of the config')

    # Instantiate model
    model = Unet(im_channels=z_channels, model_config=ldm_cfg)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    model.to(device)

    # Load checkpoint if requested or available in config
    ckpt_path = args.ckpt
    if ckpt_path is None:
        if train_cfg and 'task_name' in train_cfg and 'ldm_ckpt_name' in train_cfg:
            ckpt_path = os.path.join(train_cfg['task_name'], train_cfg['ldm_ckpt_name'])
            if not os.path.exists(ckpt_path):
                ckpt_path = None

    if ckpt_path is not None and os.path.exists(ckpt_path):
        print(f'Loading checkpoint: {ckpt_path} (mapped to {device})')
        state = torch.load(ckpt_path, map_location=device)
        try:
            model.load_state_dict(state)
        except Exception as e:
            print('Warning: failed to load full state_dict into model:', e)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('Model:', model.__class__.__name__)
    print(f'Total parameters: {total_params} ({human(total_params)})')
    print(f'Trainable parameters: {trainable_params} ({human(trainable_params)})')
    print(f'Frozen parameters: {total_params - trainable_params} ({human(total_params - trainable_params)})')

    # Optional: print top parameter-holding modules
    module_params = []
    for name, module in model.named_modules():
        # skip trivial containers
        if len(list(module.parameters(recurse=False))) == 0:
            continue
        params = sum(p.numel() for p in module.parameters(recurse=False))
        module_params.append((name or '.', params))
    module_params.sort(key=lambda x: x[1], reverse=True)

    print('\nTop modules by parameter count:')
    for name, params in module_params[:20]:
        print(f'  {name:40s} {params:12d} ({human(params)})')


if __name__ == '__main__':
    main()
