import pickle
import glob
import os
import torch


def load_latents(latent_path):
    r"""
    Simple utility to save latents to speed up ldm training
    :param latent_path:
    :return:
    """
    latent_maps = {}
    for fname in glob.glob(os.path.join(latent_path, '*.pt')):
        # Extract filename without extension
        filename_with_ext = os.path.basename(fname)
        filename_key = os.path.splitext(filename_with_ext)[0]
        
        # Convert safe filename back to original path format
        original_key = filename_key.replace('_', '/')
        
        # Load the tensor directly using torch.load
        latent_tensor = torch.load(fname, map_location='cpu')
        latent_maps[original_key] = latent_tensor
    return latent_maps


def drop_text_condition(text_embed, im, empty_text_embed, text_drop_prob):
    if text_drop_prob > 0:
        text_drop_mask = torch.zeros((im.shape[0]), device=im.device).float().uniform_(0,
                                                                                       1) < text_drop_prob
        assert empty_text_embed is not None, ("Text Conditioning required as well as"
                                        " text dropping but empty text representation not created")
        text_embed[text_drop_mask, :, :] = empty_text_embed[0]
    return text_embed


def drop_image_condition(image_condition, im, im_drop_prob):
    if im_drop_prob > 0:
        im_drop_mask = torch.zeros((im.shape[0], 1, 1, 1), device=im.device).float().uniform_(0,
                                                                                        1) > im_drop_prob
        return image_condition * im_drop_mask
    else:
        return image_condition


def drop_class_condition(class_condition, class_drop_prob, im):
    if class_drop_prob > 0:
        class_drop_mask = torch.zeros((im.shape[0], 1), device=im.device).float().uniform_(0,
                                                                                           1) > class_drop_prob
        return class_condition * class_drop_mask
    else:
        return class_condition