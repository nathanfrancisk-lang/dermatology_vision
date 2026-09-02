import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import models

# Channel statistics the torchvision ImageNet weights were trained under
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class EczemaBinaryClassificationModel(object):
    '''
    Binary classification model for eczema detection using a ResNet encoder backbone.

    Args:
        encoder_type : str
            backbone architecture: resnet18, resnet34, resnet50
        pretrained : bool
            if set, use ImageNet pretrained weights
        normalized_image_range : list[float]
            [min, max] range to normalize image intensities to
        imagenet_normalize : bool
            if set, additionally standardize by the ImageNet channel mean and std that the
            pretrained backbone expects
        device : torch.device
            device to run model on
    '''

    def __init__(self,
                 encoder_type='resnet50',
                 pretrained=True,
                 normalized_image_range=[0, 1],
                 imagenet_normalize=True,
                 device=torch.device('cuda')):

        self.encoder_type = encoder_type
        self.pretrained = pretrained
        self.normalized_image_range = normalized_image_range
        self.imagenet_normalize = imagenet_normalize
        self.device = device

        # Number of output classes for binary classification
        self.n_class = 2

        # Build the backbone
        self.model = self._build_model(encoder_type, pretrained)

    def _build_model(self, encoder_type, pretrained):
        '''
        Builds the ResNet backbone and replaces the final FC layer

        Args:
            encoder_type : str
                resnet18, resnet34, or resnet50
            pretrained : bool
                if set, use ImageNet pretrained weights
        Returns:
            nn.Module : the classification model
        '''

        weights = 'IMAGENET1K_V1' if pretrained else None

        if encoder_type == 'resnet18':
            model = models.resnet18(weights=weights)
        elif encoder_type == 'resnet34':
            model = models.resnet34(weights=weights)
        elif encoder_type == 'resnet50':
            model = models.resnet50(weights=weights)
        else:
            raise ValueError('Unsupported encoder type: {}'.format(encoder_type))

        # Replace the final fully connected layer for binary classification
        n_features = model.fc.in_features
        model.fc = nn.Linear(n_features, self.n_class)

        return model

    def forward(self, image):
        '''
        Forward pass through the model

        Args:
            image : torch.Tensor (N x C x H x W)
                input image tensor, expected in [0, 255] range
        Returns:
            torch.Tensor (N x 2) : class logits
        '''

        # Normalize image to target range
        image = self._normalize(image)

        logits = self.model(image)

        return logits

    def _normalize(self, image):
        '''
        Normalizes image intensities from [0, 255] to normalized_image_range

        Args:
            image : torch.Tensor
                input image in [0, 255]
        Returns:
            torch.Tensor : normalized image
        '''

        low, high = self.normalized_image_range

        # Scale from [0, 255] to [low, high]
        image = image / 255.0
        image = image * (high - low) + low

        # Standardize to what the pretrained backbone was trained on
        if self.imagenet_normalize:
            mean = torch.tensor(IMAGENET_MEAN, dtype=image.dtype, device=image.device)
            std = torch.tensor(IMAGENET_STD, dtype=image.dtype, device=image.device)
            image = (image - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)

        return image

    def compute_loss(self, logits, labels, loss_func, class_weights=None, label_smoothing=0.0):
        '''
        Computes the classification loss

        Args:
            logits : torch.Tensor (N x 2)
                raw model output logits
            labels : torch.Tensor (N,)
                ground truth binary labels (0 or 1)
            loss_func : str
                loss function name: cross_entropy or weighted_cross_entropy
            class_weights : list[float] or None
                class weights for weighted cross entropy
            label_smoothing : float
                label smoothing factor in [0, 1); caps how confident the model can become,
                which keeps the predicted probabilities spread out enough for the
                classification threshold to be a usable knob
        Returns:
            tuple(torch.Tensor, dict) : (scalar loss, info dict with loss breakdown)
        '''

        labels = labels.long()

        if loss_func == 'weighted_cross_entropy' and class_weights is not None:
            weight = torch.tensor(class_weights, dtype=torch.float32, device=logits.device)
            criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        loss = criterion(logits, labels)

        loss_info = {
            'loss': loss.item()
        }

        return loss, loss_info

    def parameters(self):
        '''
        Returns model parameters

        Returns:
            generator : model parameters
        '''

        return self.model.parameters()

    def named_parameters(self):
        '''
        Returns named model parameters

        Returns:
            generator : named model parameters
        '''

        return self.model.named_parameters()

    def train(self):
        '''
        Sets model to training mode
        '''

        self.model.train()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        self.model.eval()

    def to(self, device):
        '''
        Moves model to specified device

        Args:
            device : torch.device
                target device
        Returns:
            self
        '''

        self.device = device
        self.model = self.model.to(device)
        return self

    def data_parallel(self):
        '''
        Wraps the model in DataParallel for multi-GPU
        '''

        self.model = nn.DataParallel(self.model)

    def save_model(self, path, step, optimizer, preprocessing=None):
        '''
        Saves model checkpoint

        The preprocessing config travels with the weights on purpose. Inference previously
        had to re-declare the input pipeline through CLI flags, and a single missing flag
        silently fed the model a differently-shaped distribution than it was trained on.

        Args:
            path : str
                file path to save checkpoint to
            step : int
                current training step
            optimizer : torch.optim.Optimizer
                optimizer whose state will be saved
            preprocessing : dict or None
                input pipeline settings to reproduce at inference (e.g. random_crop)
        '''

        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'encoder_type': self.encoder_type,
            'normalized_image_range': self.normalized_image_range,
            'imagenet_normalize': self.imagenet_normalize,
            'preprocessing': preprocessing or {}
        }

        torch.save(checkpoint, path)

    def restore_model(self, path, optimizer=None):
        '''
        Restores model from checkpoint

        Args:
            path : str
                file path to load checkpoint from
            optimizer : torch.optim.Optimizer or None
                optimizer to restore state into
        Returns:
            tuple(int, dict) : (training step, preprocessing settings the model was trained with)
        '''

        checkpoint = torch.load(path, map_location=self.device)

        # Handle DataParallel wrapped state dicts
        state_dict = checkpoint['model_state_dict']
        model_state = self.model.state_dict()

        # Check if saved with DataParallel (keys start with 'module.')
        saved_has_module = any(k.startswith('module.') for k in state_dict.keys())
        model_has_module = any(k.startswith('module.') for k in model_state.keys())

        if saved_has_module and not model_has_module:
            # Remove 'module.' prefix
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        elif not saved_has_module and model_has_module:
            # Add 'module.' prefix
            state_dict = {'module.' + k: v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict)

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        step = checkpoint.get('step', 0)

        # Adopt the normalization the checkpoint was trained under. Checkpoints written before
        # this field existed predate ImageNet normalization, so default to off for them.
        self.imagenet_normalize = checkpoint.get('imagenet_normalize', False)

        preprocessing = checkpoint.get('preprocessing', {})

        return step, preprocessing

    def log_summary(self,
                    summary_writer,
                    tag,
                    step,
                    images=None,
                    labels=None,
                    scalars=None,
                    n_image_per_summary=4):
        '''
        Logs training/validation summary to TensorBoard

        Args:
            summary_writer : SummaryWriter
                TensorBoard summary writer
            tag : str
                tag prefix for the summary (e.g. 'train' or 'val')
            step : int
                global step number
            images : torch.Tensor or None
                batch of images to log (N x C x H x W)
            labels : torch.Tensor or None
                batch of ground truth labels (N,) aligned with images
            scalars : dict or None
                dictionary of scalar name -> value pairs to log
            n_image_per_summary : int
                max number of images to log per summary
        '''

        if summary_writer is None:
            return

        # Log scalar values
        if scalars is not None:
            for key, value in scalars.items():
                summary_writer.add_scalar('{}/{}'.format(tag, key), value, step)

        # Log sample images
        if images is not None:
            n_images = min(images.shape[0], n_image_per_summary)

            for idx in range(n_images):
                image = images[idx, ...]

                # Normalize to [0, 1] for display if needed
                if image.max() > 1.0:
                    image = image / 255.0

                if labels is not None:
                    image_tag = '{}/image_{}_label_{}'.format(tag, idx, int(labels[idx]))
                else:
                    image_tag = '{}/image_{}'.format(tag, idx)

                summary_writer.add_image(
                    image_tag,
                    image,
                    step)

        summary_writer.flush()
