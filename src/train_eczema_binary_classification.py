import argparse
from eczema_binary_classification_main import train

parser = argparse.ArgumentParser()

# Input filepaths
parser.add_argument('--train_image_paths_file',
    nargs='+', type=str, required=True, help='Paths to files containing training image paths')
parser.add_argument('--train_labels_file',
    nargs='+', type=str, required=True, help='Paths to files containing training labels')
parser.add_argument('--val_image_paths_file',
    nargs='+', type=str, required=True, help='Paths to files containing validation image paths')
parser.add_argument('--val_labels_file',
    nargs='+', type=str, required=True, help='Paths to files containing validation labels')

# Input settings
parser.add_argument('--batch_size',
    type=int, default=32, help='Batch size for training')
parser.add_argument('--n_height',
    type=int, default=224, help='Height to resize images to')
parser.add_argument('--n_width',
    type=int, default=224, help='Width to resize images to')
parser.add_argument('--normalized_image_range',
    nargs='+', type=float, default=[0, 1], help='Range to normalize image intensities to')

# Augmentation settings
parser.add_argument('--random_brightness',
    nargs='+', type=float, default=[-1, -1], help='Brightness augmentation range')
parser.add_argument('--random_contrast',
    nargs='+', type=float, default=[-1, -1], help='Contrast augmentation range')
parser.add_argument('--random_saturation',
    nargs='+', type=float, default=[-1, -1], help='Saturation augmentation range')
parser.add_argument('--random_hue',
    nargs='+', type=float, default=[-1, -1], help='Hue augmentation range')
parser.add_argument('--random_flip_type',
    nargs='+', type=str, default=['none'], help='Flip type: horizontal, vertical, none')
parser.add_argument('--random_rotate_max',
    type=float, default=0, help='Max rotation angle for augmentation')
parser.add_argument('--random_crop',
    action='store_true', help='Use aspect-ratio preserving random crop during training and center crop during validation')

# Encoder settings
parser.add_argument('--encoder_type',
    type=str, default='resnet50', help='Encoder backbone: resnet18, resnet34, resnet50')
parser.add_argument('--pretrained',
    action='store_true', help='Use ImageNet pretrained weights')
parser.add_argument('--no_imagenet_normalize',
    action='store_true', help='Skip ImageNet mean/std standardization of inputs')

# Training settings
parser.add_argument('--balance_sampler',
    action='store_true', help='Use a WeightedRandomSampler to balance training batches')
parser.add_argument('--learning_rates',
    nargs='+', type=float, default=[1e-4], help='Learning rates for the schedule')
parser.add_argument('--learning_schedule',
    nargs='+', type=int, default=[], help='Step boundaries for learning rate changes')
parser.add_argument('--learning_rate_scheduler',
    type=str, default='cosine_annealing', help='Scheduler: cosine_annealing, multi_step')
parser.add_argument('--warmup_updates',
    type=int, default=0, help='Number of warmup steps')
parser.add_argument('--warmup_init_lr',
    type=float, default=1e-7, help='Initial learning rate during warmup')
parser.add_argument('--adam_betas',
    nargs='+', type=float, default=[0.9, 0.999], help='Adam optimizer betas')
parser.add_argument('--weight_decay',
    type=float, default=1e-4, help='Weight decay for the optimizer')
parser.add_argument('--n_epoch',
    type=int, default=100, help='Number of training epochs')

# Loss settings
parser.add_argument('--loss_func',
    type=str, default='cross_entropy', help='Loss function: cross_entropy, weighted_cross_entropy')
parser.add_argument('--loss_class_weights',
    nargs='+', type=float, default=None, help='Class weights for weighted cross entropy')
parser.add_argument('--label_smoothing',
    type=float, default=0.0, help='Label smoothing factor in [0, 1); counters overconfidence')

# Evaluation settings
parser.add_argument('--classification_threshold',
    type=float, default=0.5, help='Probability threshold on P(eczema) for positive prediction; lower favors recall')
parser.add_argument('--target_sensitivity',
    type=float, default=0.9, help='Target eczema sensitivity; selects checkpoints and calibrates the saved threshold')

# Checkpoint and summary settings
parser.add_argument('--checkpoint_dirpath',
    type=str, required=True, help='Directory for saving checkpoints')
parser.add_argument('--n_step_per_checkpoint',
    type=int, default=1000, help='Steps between checkpoints')
parser.add_argument('--n_step_per_summary',
    type=int, default=100, help='Steps between tensorboard summaries')
parser.add_argument('--n_display',
    type=int, default=5, help='Number of display samples per summary')
parser.add_argument('--start_step_validation',
    type=int, default=0, help='Step to start running validation')
parser.add_argument('--early_stop_patience',
    type=int, default=0, help='Stop after this many validations without improvement; 0 disables')
parser.add_argument('--restore_path',
    type=str, default=None, help='Path to restore model checkpoint from')

# Hardware settings
parser.add_argument('--device',
    type=str, default='gpu', help='Device to use: gpu, cpu')
parser.add_argument('--n_thread',
    type=int, default=8, help='Number of data loading threads')

args = parser.parse_args()

if __name__ == '__main__':
    train(train_image_paths_file=args.train_image_paths_file,
          train_labels_file=args.train_labels_file,
          val_image_paths_file=args.val_image_paths_file,
          val_labels_file=args.val_labels_file,
          # Input settings
          batch_size=args.batch_size,
          n_height=args.n_height,
          n_width=args.n_width,
          normalized_image_range=args.normalized_image_range,
          # Augmentation settings
          random_brightness=args.random_brightness,
          random_contrast=args.random_contrast,
          random_saturation=args.random_saturation,
          random_hue=args.random_hue,
          random_flip_type=args.random_flip_type,
          random_rotate_max=args.random_rotate_max,
          random_crop=args.random_crop,
          # Encoder settings
          encoder_type=args.encoder_type,
          pretrained=args.pretrained,
          imagenet_normalize=not args.no_imagenet_normalize,
          # Training settings
          balance_sampler=args.balance_sampler,
          learning_rates=args.learning_rates,
          learning_schedule=args.learning_schedule,
          learning_rate_scheduler=args.learning_rate_scheduler,
          warmup_updates=args.warmup_updates,
          warmup_init_lr=args.warmup_init_lr,
          adam_betas=args.adam_betas,
          weight_decay=args.weight_decay,
          n_epoch=args.n_epoch,
          # Loss settings
          loss_func_name=args.loss_func,
          loss_class_weights=args.loss_class_weights,
          label_smoothing=args.label_smoothing,
          # Evaluation settings
          classification_threshold=args.classification_threshold,
          target_sensitivity=args.target_sensitivity,
          # Checkpoint settings
          checkpoint_dirpath=args.checkpoint_dirpath,
          n_step_per_checkpoint=args.n_step_per_checkpoint,
          n_step_per_summary=args.n_step_per_summary,
          n_display=args.n_display,
          start_step_validation=args.start_step_validation,
          early_stop_patience=args.early_stop_patience,
          restore_path=args.restore_path,
          # Hardware settings
          device=args.device,
          n_thread=args.n_thread)