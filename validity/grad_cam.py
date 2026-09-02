'''
Grad-CAM overlays for the eczema binary classifier.

Answers a question the metrics cannot: whether a correct prediction came from the lesion or from
something incidental that happens to correlate with it. Dermatology datasets are full of such
shortcuts - rulers, ink markings, hair, clinical backgrounds, body-part framing - and a model
that reads those scores well on a test set drawn from the same source and fails everywhere else.

Reads the same checkpoint and the same deterministic preprocessing as run_eczema_binary_classification.py,
so the heatmaps correspond to the predictions being reported rather than to a re-processed image.

Run:  python src/grad_cam.py --image_paths_file ... --labels_file ... --checkpoint_path ...
'''

import argparse, os, sys
import cv2
import numpy as np
import torch

sys.path.insert(0, 'src')
import data_utils
import datasets
from eczema_binary_classification_model import EczemaBinaryClassificationModel


parser = argparse.ArgumentParser()
parser.add_argument('--image_paths_file',
    type=str, required=True, help='File containing image paths')
parser.add_argument('--labels_file',
    type=str, required=True, help='File containing binary labels')
parser.add_argument('--checkpoint_path',
    type=str, required=True, help='Path to model checkpoint')
parser.add_argument('--probabilities_file',
    type=str, default=None, help='probabilities.txt from a prior inference run; avoids recomputing scores to rank cases')
parser.add_argument('--threshold_path',
    type=str, default=None, help='File holding the calibrated decision threshold')
parser.add_argument('--classification_threshold',
    type=float, default=0.5, help='Decision threshold, overridden by --threshold_path')
parser.add_argument('--n_height',
    type=int, default=224, help='Height images were trained at')
parser.add_argument('--n_width',
    type=int, default=224, help='Width images were trained at')
parser.add_argument('--normalized_image_range',
    nargs='+', type=float, default=[0, 1], help='Range to normalize image intensities to')
parser.add_argument('--encoder_type',
    type=str, default='resnet50', help='Encoder backbone')
parser.add_argument('--n_sample',
    type=int, default=6, help='Number of cases to render per category')
parser.add_argument('--output_path',
    type=str, required=True, help='Directory to write the overlays to')
parser.add_argument('--device',
    type=str, default='gpu', help='Device to use: gpu, cpu')


def compute_grad_cam(model, image, target_class=1):
    '''
    Computes a Grad-CAM heatmap for one image

    Weights each channel of the last convolutional block by the gradient of the target logit with
    respect to that channel, sums, and keeps the positive part. layer4 is the deepest layer that
    still carries spatial structure, so it localizes without being purely semantic.

    Args:
        model : EczemaBinaryClassificationModel
            restored model, already on the target device
        image : torch.Tensor (1 x C x H x W)
            input image in [0, 255]
        target_class : int
            logit to explain; 1 is eczema
    Returns:
        numpy[float] (H x W) : heatmap in [0, 1] at the input resolution
    '''

    activations, gradients = {}, {}

    # resnet layer4 is the last conv block before global pooling
    target_layer = model.model.layer4

    # Both hooks must return None. A forward hook that returns a value replaces the layer output,
    # and a backward hook that returns one replaces grad_input, which torch rejects on a shape
    # mismatch. Capturing into a dict and returning nothing keeps the graph untouched.
    def save_activation(module, inputs, output):
        activations['value'] = output

    def save_gradient(module, grad_input, grad_output):
        gradients['value'] = grad_output[0]

    forward_handle = target_layer.register_forward_hook(save_activation)
    backward_handle = target_layer.register_full_backward_hook(save_gradient)

    try:
        # No no_grad here: the gradient through the activations is the whole method
        logits = model.forward(image)
        model.model.zero_grad()
        logits[0, target_class].backward()

        activation = activations['value'][0]
        gradient = gradients['value'][0]

        # One weight per channel, averaged over space
        weights = gradient.mean(dim=(1, 2))
        cam = torch.relu((weights.view(-1, 1, 1) * activation).sum(dim=0))
    finally:
        forward_handle.remove()
        backward_handle.remove()

    cam = cam.detach().cpu().numpy()
    cam = cv2.resize(cam, (image.shape[3], image.shape[2]))

    # Normalizing per image keeps low-confidence cases legible; it also means brightness is not
    # comparable between overlays, only the spatial pattern within one.
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())

    return cam

def overlay_heatmap(image, cam):
    '''
    Blends a heatmap over an image, side by side with the original

    Args:
        image : numpy[float] (C x H x W)
            input image in [0, 255]
        cam : numpy[float] (H x W)
            heatmap in [0, 1]
    Returns:
        numpy[uint8] (H x 2W x 3) : original beside the overlay, RGB
    '''

    image = np.transpose(image, (1, 2, 0)).astype(np.uint8)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    blended = np.uint8(0.6 * image + 0.4 * heatmap)

    return np.concatenate([image, blended], axis=1)


if __name__ == '__main__':
    args = parser.parse_args()

    device = torch.device('cuda' if args.device == 'gpu' and torch.cuda.is_available() else 'cpu')

    model = EczemaBinaryClassificationModel(
        encoder_type=args.encoder_type,
        pretrained=False,
        normalized_image_range=args.normalized_image_range,
        device=device)

    _, preprocessing = model.restore_model(args.checkpoint_path)
    model.to(device)
    model.eval()

    # Same crop geometry as training and inference, read off the checkpoint rather than assumed
    dataset = datasets.EczemaDataset(
        dataset='test',
        image_paths=args.image_paths_file,
        labels=args.labels_file,
        shape=(args.n_height, args.n_width),
        random_crop=preprocessing.get('random_crop', False))

    labels = np.array(dataset.labels)

    threshold = args.classification_threshold
    if args.threshold_path is not None:
        with open(args.threshold_path, 'r') as f:
            threshold = float(f.read().strip())

    # EczemaDataset yields numpy arrays; batching into a tensor is normally the DataLoader's job
    as_batch = lambda image: torch.from_numpy(image).unsqueeze(0).to(device)

    if args.probabilities_file is not None:
        probabilities = np.loadtxt(args.probabilities_file, dtype=np.float32, ndmin=1)
        assert len(probabilities) == len(labels), \
            '{} probabilities vs {} labels'.format(len(probabilities), len(labels))
    else:
        probabilities = np.zeros(len(labels), dtype=np.float32)
        with torch.no_grad():
            for idx in range(len(dataset)):
                image, _ = dataset[idx]
                logits = model.forward(as_batch(image))
                probabilities[idx] = torch.softmax(logits, dim=1)[0, 1].item()

    predictions = (probabilities >= threshold).astype(np.int64)

    # Both the cases that worked and the cases that did not. Showing only confident hits proves
    # nothing; the false positives are where a shortcut feature shows itself.
    categories = {
        'true_positive':  np.where((predictions == 1) & (labels == 1))[0],
        'false_positive': np.where((predictions == 1) & (labels == 0))[0],
        'false_negative': np.where((predictions == 0) & (labels == 1))[0]
    }

    os.makedirs(args.output_path, exist_ok=True)

    for category, indices in categories.items():
        if len(indices) == 0:
            print('{}: none in this set'.format(category))
            continue

        # Rank by how strongly the model committed, so these are the cases it was surest about
        ranked = indices[np.argsort(-probabilities[indices])]
        if category == 'false_negative':
            ranked = indices[np.argsort(probabilities[indices])]

        for rank, idx in enumerate(ranked[:args.n_sample]):
            image, _ = dataset[idx]
            cam = compute_grad_cam(model, as_batch(image), target_class=1)

            output_filepath = os.path.join(
                args.output_path, '{}_{}_p{:.3f}.png'.format(category, rank, probabilities[idx]))

            data_utils.save_image(
                image=overlay_heatmap(image, cam),
                path=output_filepath,
                normalized=False,
                data_type='color',
                data_format='HWC')

        print('{}: wrote {} overlays ({} cases available)'.format(
            category, min(args.n_sample, len(ranked)), len(indices)))

    print('\nLeft half of each image is the input, right half is the Grad-CAM overlay.')
    print('Wrote to {}'.format(args.output_path))
