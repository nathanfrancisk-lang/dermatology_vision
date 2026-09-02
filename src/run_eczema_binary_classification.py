import argparse
from eczema_binary_classification_main import run

parser = argparse.ArgumentParser()

# Input filepaths
parser.add_argument('--image_paths_file',
    nargs='+', type=str, required=True, help='Paths to files containing image paths')
parser.add_argument('--labels_file',
    nargs='+', type=str, required=True, help='Paths to files containing labels')
parser.add_argument('--checkpoint_path',
    type=str, required=True, help='Path to model checkpoint')

# Input settings
parser.add_argument('--batch_size',
    type=int, default=32, help='Batch size for inference')
parser.add_argument('--n_height',
    type=int, default=224, help='Image height')
parser.add_argument('--n_width',
    type=int, default=224, help='Image width')
parser.add_argument('--normalized_image_range',
    nargs='+', type=float, default=[0, 1], help='Range to normalize image intensities to')
parser.add_argument('--random_crop',
    action='store_true', default=None,
    help='Force aspect-preserving scale and center crop; by default follow the checkpoint')

# Encoder settings
parser.add_argument('--encoder_type',
    type=str, default='resnet50', help='Encoder backbone: resnet18, resnet34, resnet50')

# Evaluation settings
parser.add_argument('--classification_threshold',
    type=float, default=0.5, help='Probability threshold on P(eczema) for positive prediction; lower favors recall')
parser.add_argument('--threshold_path',
    type=str, default=None, help='Path to threshold.txt written by training; overrides --classification_threshold')

# Output settings
parser.add_argument('--output_path',
    type=str, default=None, help='Directory to save results')
parser.add_argument('--save_outputs',
    action='store_true', help='Whether to save prediction outputs')

# Hardware settings
parser.add_argument('--device',
    type=str, default='gpu', help='Device to use: gpu, cpu')
parser.add_argument('--n_thread',
    type=int, default=8, help='Number of data loading threads')

args = parser.parse_args()

if __name__ == '__main__':
    run(image_paths_file=args.image_paths_file,
        labels_file=args.labels_file,
        checkpoint_path=args.checkpoint_path,
        # Input settings
        batch_size=args.batch_size,
        n_height=args.n_height,
        n_width=args.n_width,
        normalized_image_range=args.normalized_image_range,
        random_crop=args.random_crop,
        # Encoder settings
        encoder_type=args.encoder_type,
        # Evaluation settings
        classification_threshold=args.classification_threshold,
        threshold_path=args.threshold_path,
        # Output settings
        output_path=args.output_path,
        save_outputs=args.save_outputs,
        # Hardware settings
        device=args.device,
        n_thread=args.n_thread)