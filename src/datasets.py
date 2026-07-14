import os, sys, argparse
import torch
import numpy as np
from torch.utils.data import Dataset
import data_utils
import cv2
from data_utils import resize, random_crop


class EczemaDataset(Dataset):
    '''
    Returns image and binary label in style of output format

    Args:
        dataset : str
            Dataset name (e.g. 'dermnet', 'fitzpatrick17k', 'sd198')
        image_paths : list[str] or str
            List of image paths or file containing image paths
        labels : list[int] or str
            List of binary labels or file containing binary labels
        shape : tuple[int, int]
            Target shape (height, width) for resizing
    '''
    def __init__(self,
                 dataset,
                 image_paths,
                 labels,
                 shape=None,
                 random_crop=False):
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

        inputs = [image]

        if self.labels[index] is not None:
            # Wrap label in a numpy array to match inputs list pattern
            label = np.array(self.labels[index], dtype=np.float32)
            inputs.append(label)

        # Convert to float32
        inputs = [T.astype(np.float32) for T in inputs]

        return inputs

    def __len__(self):
        '''
        Returns the number of elements in dataset

        Returns:
            int : number of elements in dataset
        '''

        return self.n_sample
