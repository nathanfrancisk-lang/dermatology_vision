import os, sys, argparse
import torch
import numpy as np
from torch.utils.data import Dataset
import data_utils
import cv2
from data_utils import resize, random_crop
from torchvision import transforms
import torchvision.transforms.functional as TF


def _is_enabled(value_range):
    '''
    Returns whether an augmentation range is active

    The CLI uses [-1, -1] as the "off" sentinel for these ranges.

    Args:
        value_range : list[float] or None
            [min, max] augmentation range
    Returns:
        bool : True if the range should be applied
    '''

    return value_range is not None and len(value_range) == 2 and value_range != [-1, -1]


class EczemaDataset(Dataset):
    '''
    Returns image and binary label in style of output format

    Augmentation is applied only when dataset == 'train'. Validation and test are fully
    deterministic: no photometric jitter, no flips, no rotation, and a center rather than
    random crop, so repeated evaluation of the same checkpoint returns identical numbers.

    Args:
        dataset : str
            Split role: 'train' gets augmentation and random crops, anything else is
            deterministic
        image_paths : list[str] or str
            List of image paths or file containing image paths
        labels : list[int] or str
            List of binary labels or file containing binary labels
        shape : tuple[int, int]
            Target shape (height, width) for resizing
        random_crop : bool
            If set, scale to 1.15x the target and crop, preserving aspect ratio
        random_brightness : list[float]
            [min, max] brightness multiplier range, [-1, -1] to disable
        random_contrast : list[float]
            [min, max] contrast multiplier range, [-1, -1] to disable
        random_saturation : list[float]
            [min, max] saturation multiplier range, [-1, -1] to disable
        random_hue : list[float]
            [min, max] hue shift range in [-0.5, 0.5], [-1, -1] to disable
        random_flip_type : list[str]
            Any of 'horizontal', 'vertical', or 'none'
        random_rotate_max : float
            Maximum rotation in degrees, 0 to disable
    '''
    def __init__(self,
                 dataset,
                 image_paths,
                 labels,
                 shape=None,
                 random_crop=False,
                 random_brightness=[-1, -1],
                 random_contrast=[-1, -1],
                 random_saturation=[-1, -1],
                 random_hue=[-1, -1],
                 random_flip_type=['none'],
                 random_rotate_max=0):
        self.dataset = dataset

        # Support loading directly from files or lists
        if isinstance(image_paths, str):
            self.image_paths = data_utils.read_paths(image_paths)
        else:
            self.image_paths = image_paths

        self.n_sample = len(self.image_paths)

        if labels is not None:
            if isinstance(labels, str):
                label_strings = data_utils.read_paths(labels)
                self.labels = [int(lbl) for lbl in label_strings]
            else:
                self.labels = [int(lbl) for lbl in labels]
        else:
            self.labels = [None] * self.n_sample

        assert self.n_sample == len(self.labels)

        self.shape = shape
        self.random_crop = random_crop

        # Shape is not None and it does not contain None
        self.do_resize = self.shape is not None and None not in self.shape

        self.data_format = 'CHW'

        # Augmentation applies to the training split only
        self.do_augment = self.dataset == 'train'

        self.random_flip_type = [f.lower() for f in (random_flip_type or ['none'])]
        self.random_rotate_max = random_rotate_max or 0

        # ColorJitter takes (min, max) tuples and treats a missing arg as "leave alone"
        jitter_kwargs = {}

        if _is_enabled(random_brightness):
            jitter_kwargs['brightness'] = tuple(random_brightness)
        if _is_enabled(random_contrast):
            jitter_kwargs['contrast'] = tuple(random_contrast)
        if _is_enabled(random_saturation):
            jitter_kwargs['saturation'] = tuple(random_saturation)
        if _is_enabled(random_hue):
            jitter_kwargs['hue'] = tuple(random_hue)

        self.color_jitter = transforms.ColorJitter(**jitter_kwargs) if jitter_kwargs else None

    def __getitem__(self, index):

        image = data_utils.load_image(
            self.image_paths[index],
            normalize=False,
            data_format=self.data_format)

        if self.do_resize:
            if self.random_crop:
                th, tw = self.shape
                _, H, W = image.shape
                
                # Scale so that the shorter dimension is 1.15x the target dimension
                scale = max(th / H, tw / W) * 1.15
                nh = int(round(H * scale))
                nw = int(round(W * scale))
                
                image = resize(image, (nh, nw), data_format=self.data_format)
                
                if self.dataset == 'train':
                    # Random crop in both dimensions
                    image = random_crop([image], (th, tw), crop_type=['horizontal', 'vertical'])[0]
                else:
                    # Center crop
                    image = random_crop([image], (th, tw), crop_type=['none'])[0]
            else:
                # Direct resize (original behavior)
                image = resize(image, self.shape, data_format=self.data_format)

        if self.do_augment:
            image = self._augment(image)

        inputs = [image]

        if self.labels[index] is not None:
            # Wrap label in a numpy array to match inputs list pattern
            label = np.array(self.labels[index], dtype=np.float32)
            inputs.append(label)

        # Convert to float32
        inputs = [T.astype(np.float32) for T in inputs]

        return inputs

    def _augment(self, image):
        '''
        Applies photometric and geometric augmentation to a training image

        Args:
            image : numpy[float32]
                C x H x W image in [0, 255]
        Returns:
            numpy[float32] : augmented C x H x W image in [0, 255]
        '''

        # torchvision's hue adjustment requires uint8 or [0, 1] floats, so round-trip
        # through uint8 and hand the model back the [0, 255] floats it expects
        tensor = torch.from_numpy(np.ascontiguousarray(image)).clamp(0, 255).to(torch.uint8)

        if self.color_jitter is not None:
            tensor = self.color_jitter(tensor)

        # Lesion orientation carries no diagnostic meaning, so both axes are fair game
        if 'horizontal' in self.random_flip_type and torch.rand(1).item() > 0.5:
            tensor = TF.hflip(tensor)

        if 'vertical' in self.random_flip_type and torch.rand(1).item() > 0.5:
            tensor = TF.vflip(tensor)

        if self.random_rotate_max > 0:
            angle = float(torch.empty(1).uniform_(
                -self.random_rotate_max, self.random_rotate_max).item())
            # ponytail: rotating after the crop leaves black corners; rotate before the
            # 1.15x crop instead if those corners show up as a learned artifact
            tensor = TF.rotate(tensor, angle)

        return tensor.numpy().astype(np.float32)

    def __len__(self):
        '''
        Returns the number of elements in dataset

        Returns:
            int : number of elements in dataset
        '''

        return self.n_sample
