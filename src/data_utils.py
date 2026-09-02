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

def stratified_split(strata, n_split, seed=13):
    '''
    Splits indices into groups holding the composition of strata roughly constant

    Used by the setup scripts to carve validation and test splits at dataset creation time, so
    no downstream script has to re-derive them. Stratifying matters at these class ratios: eczema
    is under 13% of every split, so an unstratified draw can shift the positive rate by a couple
    of points, which moves any sensitivity target evaluated against it.

    Arg(s):
        strata : list
            per-sample grouping key; samples sharing a key are divided in the same proportions
        n_split : list[float]
            fractions to divide into, must sum to 1
        seed : int
            seed for the shuffle, fixed so splits are reproducible across runs
    Return:
        list[list[int]] : one sorted index list per entry in n_split
    '''

    assert abs(sum(n_split) - 1.0) < 1e-9, \
        'n_split must sum to 1, got {}'.format(sum(n_split))

    rng = random.Random(seed)
    splits = [[] for _ in n_split]

    for stratum in sorted(set(strata), key=str):
        indices = [idx for idx, s in enumerate(strata) if s == stratum]
        rng.shuffle(indices)

        # Largest-remainder allocation: floor every share, then hand out the leftovers to the
        # splits with the largest fractional parts. Rounding each share independently can drop
        # or duplicate a sample when a stratum is small, and the rare strata here (Fitzpatrick
        # type VI) are exactly the ones worth counting correctly.
        exact = [len(indices) * fraction for fraction in n_split]
        counts = [int(value) for value in exact]

        leftover = len(indices) - sum(counts)
        ranked = sorted(range(len(n_split)), key=lambda i: exact[i] - counts[i], reverse=True)
        for split_idx in ranked[:leftover]:
            counts[split_idx] += 1

        offset = 0
        for split_idx, count in enumerate(counts):
            splits[split_idx].extend(indices[offset:offset + count])
            offset += count

    return [sorted(split) for split in splits]

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

def resize(T, shape, interp_type='lanczos', data_format='HWC', lib_type='cv2'):
    '''
    Resizes a tensor
    Args:
        T : numpy
            tensor to resize
        shape : tuple[int]
            (height, width) to resize tensor
        interp_type : str
            interpolation for resize
        data_format : str
            'CHW', or 'HWC', 'CDHW', 'DHWC'
    Returns:
        numpy : image resized to height and width
    '''
    if shape is None or any([x is None or x <= 0 for x in shape]):
        return T

    n_height, n_width = shape

    if lib_type == 'cv2':
        resize_shape = (n_width, n_height)

        resize_func = cv2.resize

        if interp_type == 'nearest':
            interp_type = cv2.INTER_NEAREST
        elif interp_type == 'area':
            interp_type = cv2.INTER_AREA
        elif interp_type == 'bilinear':
            interp_type = cv2.INTER_LINEAR
        elif interp_type == 'lanczos':
            interp_type = cv2.INTER_LANCZOS4
        else:
            raise ValueError('Unsupport interpolation type: {} for library {}'.format(interp_type, lib_type))

    elif lib_type == 'pil':
        resize_shape = (n_height, n_width)

        def pil_resize(R, shape, interpolation=None):
            R = Image.fromarray(np.uint8(R))

            # Resize and transpose back to CHW
            R = transforms.functional.resize(R, shape, interpolation=interpolation)

            R = np.array(R).astype(np.float32)

            return R

        resize_func = pil_resize

        if interp_type == 'nearest':
            interp_type = transforms.InterpolationMode.NEAREST
        elif interp_type == 'bilinear':
            interp_type = transforms.InterpolationMode.BILINEAR
        elif interp_type == 'lanczos':
            interp_type = transforms.InterpolationMode.LANCZOS
        else:
            raise ValueError('Unsupport interpolation type: {} for library {}'.format(interp_type, lib_type))

    # Resize tensor
    if data_format == 'CHW':
        # Tranpose from CHW to HWC
        R = np.transpose(T, (1, 2, 0))

        # Resize and transpose back to CHW
        R = resize_func(R, resize_shape, interpolation=interp_type)

        R = np.reshape(R, (n_height, n_width, T.shape[0]))
        R = np.transpose(R, (2, 0, 1))

    elif data_format == 'HWC':
        R = resize_func(T, resize_shape, interpolation=interp_type)
        R = np.reshape(R, (n_height, n_width, T.shape[2]))

    elif data_format == 'CDHW':
        # Transpose CDHW to DHWC
        D = np.transpose(T, (1, 2, 3, 0))

        # Resize and transpose back to CDHW
        R = np.zeros((D.shape[0], n_height, n_width, D.shape[3]))

        for d in range(R.shape[0]):
            r = resize_func(D[d, ...], resize_shape, interpolation=interp_type)
            R[d, ...] = np.reshape(r, (n_height, n_width, D.shape[3]))

        R = np.transpose(R, (3, 0, 1, 2))

    elif data_format == 'DHWC':
        R = np.zeros((T.shape[0], n_height, n_width, T.shape[3]))
        for d in range(R.shape[0]):
            r = resize_func(T[d, ...], resize_shape, interpolation=interp_type)
            R[d, ...] = np.reshape(r, (n_height, n_width, T.shape[3]))

    else:
        raise ValueError('Unsupport data format: {}'.format(data_format))

    return R

def random_crop(inputs, shape, intrinsics=None, crop_type=['none']):
    '''
    Apply crop to inputs e.g. images, depth and if available adjust camera intrinsics

    Arg(s):
        inputs : list[numpy[float32]]
            list of numpy arrays e.g. images, depth, and validity maps
        shape : list[int]
            shape (height, width) to crop inputs
        intrinsics : list[numpy[float32]]
            list of 3 x 3 camera intrinsics matrix
        crop_type : str
            none, horizontal, vertical, anchored, bottom
    Return:
        list[numpy[float32]] : list of cropped inputs
        list[numpy[float32]] : if given, 3 x 3 adjusted camera intrinsics matrix
    '''

    n_height, n_width = shape
    _, o_height, o_width = inputs[0].shape

    n_height = o_height if n_height == 0 else n_height
    n_width = o_width if n_width == 0 else n_width

    # Get delta of crop and original height and width
    d_height = o_height - n_height
    d_width = o_width - n_width

    # By default, perform center crop
    y_start = d_height // 2
    x_start = d_width // 2

    if 'horizontal' in crop_type:

        # Select from one of the pre-defined anchored locations
        if 'anchored' in crop_type:
            # Create anchor positions
            crop_anchors = [
                0.0, 0.50, 1.0
            ]

            widths = [
                anchor * d_width for anchor in crop_anchors
            ]
            x_start = int(widths[np.random.randint(low=0, high=len(widths))])

        # Randomly select a crop location
        else:
            x_start = np.random.randint(low=0, high=d_width+1)

    # If bottom alignment, then set starting height to bottom position
    if 'bottom' in crop_type:
        y_start = d_height

    elif 'vertical' in crop_type:

        # Select from one of the pre-defined anchored locations
        if 'anchored' in crop_type:
            # Create anchor positions
            crop_anchors = [
                0.0, 0.50, 1.0
            ]

            heights = [
                anchor * d_height for anchor in crop_anchors
            ]
            y_start = int(heights[np.random.randint(low=0, high=len(heights))])

        # Randomly select a crop location
        else:
            y_start = np.random.randint(low=0, high=d_height+1)

    # Crop each input into (n_height, n_width)
    y_end = y_start + n_height
    x_end = x_start + n_width

    outputs = [
        T[:, y_start:y_end, x_start:x_end] for T in inputs
    ]

    # Adjust intrinsics
    if intrinsics is not None:
        offset_principal_point = [[0.0, 0.0, -x_start],
                                  [0.0, 0.0, -y_start],
                                  [0.0, 0.0, 0.0     ]]

        intrinsics = [
            in_ + offset_principal_point for in_ in intrinsics
        ]

        return outputs, intrinsics
    else:
        return outputs