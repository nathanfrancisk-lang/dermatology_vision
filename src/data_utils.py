import cv2
import numpy as np
from PIL import Image
import torch
import re, os, sys
from collections import Counter
import random
from torchvision import transforms


def read_paths(filepath):
    '''
    Reads a newline delimited file containing paths

    Arg(s):
        filepath : str
            path to file to be read
    Return:
        list[str] : list of paths
    '''

    path_list = []
    with open(filepath) as f:
        while True:
            path = f.readline().rstrip()

            # If there was nothing to read
            if path == '':
                break

            path_list.append(path)

    return path_list

def write_paths(filepath, paths):
    '''
    Stores line delimited paths into file

    Arg(s):
        filepath : str
            path to file to save paths
        paths : list[str]
            paths to write into file
    '''

    with open(filepath, 'w') as o:
        for idx in range(len(paths)):
            o.write(paths[idx] + '\n')

def load_image(path, normalize=False, data_type='color', data_format='HWC'):
    '''
    Loads an RGB, gray, or label (with validity map in alpha) image

    Args:
        path : str
            path to RGB, gray or label (with validity map in alpha) image
        normalize : bool
            if set, then normalize image between [0, 1]
        data_type : str
            color, gray or label
        data_format : str
            'CHW', or 'HWC'
    Returns:
        numpy : H x W x C or C x H x W image
    '''

    # Load image
    if data_type == 'color':
        image = Image.open(path).convert('RGB')
    elif data_type == 'gray':
        image = Image.open(path).convert('L')
    elif data_type == 'label':
        image = Image.open(path).convert('LA')
    else:
        raise ValueError('Unsupported data type: {}'.format(data_type))

    # Convert to numpy
    image = np.asarray(image, np.float32)

    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)

    if data_format == 'HWC':
        pass
    elif data_format == 'CHW':
        image = np.transpose(image, (2, 0, 1))
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

    # Normalize
    image = image / 255.0 if normalize else image

    return image

def save_image(image, path, normalized=True, data_type='color', data_format='HWC'):
    '''
    Saves an RGB, gray, or label (with validity map in alpha) image to 8-bit PNG

    Arg(s):
        image : numpy
            RGB, gray, or label (with validity map in alpha) image
        path : str
            path to store image
        normalized : bool
            if set, then treat image as normalized range [0, 1] and multiply by 255
        data_type : str
            color or label
        data_format : str
            data format of input 'CHW', or 'HWC'
    '''

    # Put image between [0, 255] range, if it was normalized to [0, 1]
    image = 255.0 * image if normalized else image
    image = np.uint8(image)

    if image.ndim == 2:
        # Append channel dimension
        image = np.expand_dims(image, axis=-1)

    # Put image into H x W x C format
    if data_format == 'HWC':
        pass
    elif data_format == 'CHW':
        image = np.transpose(image, (1, 2, 0))
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

    if data_type == 'color':
        image = Image.fromarray(np.uint8(image))
    elif data_type == 'gray':
        image = np.squeeze(image)
        image = Image.fromarray(np.uint8(image), mode='L')
    elif data_type == 'label':
        # Add alpha channel
        if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
            image = np.concatenate([image, 255.0 * np.ones_like(image)], axis=-1)
        elif image.ndim == 3 and image.shape[-1] > 2:
            raise ValueError('ERROR: too many channels in label')

        image = Image.fromarray(np.uint8(image), mode='LA')
    else:
        raise ValueError('Unsupported data type: {}'.format(data_type))

    image.save(path)