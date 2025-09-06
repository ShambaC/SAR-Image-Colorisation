"""Utility to count model parameters for the Diffusion U-Net.
"""
from __future__ import annotations
import argparse
import os
import sys
from collections import defaultdict

import torch

# Adjust path so this script can be run from repository root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from diffusion import Diffusion


def count_params(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def breakdown_by_top_level(model: torch.nn.Module):
    counts = defaultdict(lambda: {'total': 0, 'trainable': 0})
    for name, p in model.named_parameters():
        top = name.split('.')[0]
        counts[top]['total'] += p.numel()
        if p.requires_grad:
            counts[top]['trainable'] += p.numel()
    return counts


def try_load_checkpoint(diffusion: Diffusion, ckpt_path: str, device: str):
    """Try several sensible ways to load weights into the Diffusion model.
    Returns (success: bool, message: str)
    """
    # Prefer the converter for .ckpt files (Stable Diffusion style)
    try:
        if ckpt_path.endswith('.ckpt') or '.ckpt' in ckpt_path:
            import model_converter
            print(f"Loading and converting checkpoint via model_converter: {ckpt_path}")
            converted = model_converter.load_from_standard_weights(ckpt_path, device)
            if 'diffusion' in converted:
                state = converted['diffusion']
            else:
                state = converted
            try:
                diffusion.load_state_dict(state, strict=True)
                return True, 'Loaded via model_converter (strict=True)'
            except Exception as e:
                # try non-strict
                diffusion.load_state_dict(state, strict=False)
                return True, f'Loaded via model_converter (strict=False) — {e}'

        # Otherwise attempt torch.load for .pt/.pth saved state dicts
        data = torch.load(ckpt_path, map_location=device)
        # If the file is a dict containing submodules like 'diffusion'
        if isinstance(data, dict) and 'diffusion' in data:
            state = data['diffusion']
            try:
                diffusion.load_state_dict(state, strict=True)
                return True, "Loaded 'diffusion' key from checkpoint (strict=True)"
            except Exception as e:
                diffusion.load_state_dict(state, strict=False)
                return True, f"Loaded 'diffusion' key from checkpoint (strict=False) — {e}"

        # If the dict looks like a state_dict for the model directly
        if isinstance(data, dict):
            try:
                diffusion.load_state_dict(data, strict=True)
                return True, "Loaded checkpoint as direct state_dict (strict=True)"
            except Exception as e:
                try:
                    diffusion.load_state_dict(data, strict=False)
                    return True, f"Loaded checkpoint as direct state_dict (strict=False) — {e}"
                except Exception:
                    pass

        return False, 'Checkpoint format not recognised or could not be loaded.'
    except Exception as e:
        return False, f'Error while loading checkpoint: {e}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', '-c', type=str, default=None, help='Path to checkpoint (.ckpt, .pt, .pth)')
    parser.add_argument('--device', '-d', type=str, default='cuda', help='Device to map weights to (cpu or cuda)')
    parser.add_argument('--no-load', action='store_true', help="Don't attempt to load a checkpoint; just instantiate the model")
    args = parser.parse_args()

    device = args.device
    if device != 'cpu' and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to cpu")
        device = 'cpu'

    print(f"Instantiating Diffusion model on device={device}...")
    model = Diffusion()
    model.to(device)

    loaded = False
    load_msg = 'No checkpoint loaded.'
    if args.ckpt and not args.no_load:
        success, msg = try_load_checkpoint(model, args.ckpt, device)
        loaded = success
        load_msg = msg
    elif not args.no_load and not args.ckpt:
        print("No --ckpt provided, skipping load. Use --ckpt <path> to load weights or --no-load to force not loading.")

    print(load_msg)

    total, trainable = count_params(model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Trainable percentage: {100.0 * trainable / total:.2f}%")

    print('\nTop-level breakdown:')
    for top, vals in breakdown_by_top_level(model).items():
        print(f" - {top}: total={vals['total']:,}, trainable={vals['trainable']:,}")


if __name__ == '__main__':
    main()
