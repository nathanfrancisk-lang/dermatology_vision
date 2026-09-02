import os, sys
from collections import Counter
import csv
import numpy as np
from PIL import Image
sys.path.insert(0, 'src')
import data_utils
import eczema_labels
import argparse
from tqdm import tqdm

# Parse arguments globally
parser = argparse.ArgumentParser(description="Process the Fitzpatrick17k dataset for Eczema binary classification.")

parser.add_argument("--root_dir", type=str, required=True, help="Path to the raw Fitzpatrick17k dataset directory (contains data/finalfitz17k/ and fitzpatrick17k.csv).")
parser.add_argument("--output_dir", type=str, help="Path to save the derived dataset and images. Required unless --splits_only.")
parser.add_argument("--splits_only", action="store_true", help="Rebuild the train/test split files from the existing reference files, skipping image processing.")

args = parser.parse_args()

TRAIN_REF_DIRPATH = os.path.join('training', 'fitzpatrick17k')
TEST_REF_DIRPATH = os.path.join('testing', 'fitzpatrick17k')

# Fitzpatrick17k is the only dataset here carrying skin tone, so it is the only place a
# tone-stratified test set can come from. Holding out 20% costs ~3300 training images and buys
# sensitivity broken out by Fitzpatrick type, which is the first thing a dermatology audience
# asks about. Stratifying on (skin tone, eczema) jointly keeps type VI (635 images) and the 4%
# eczema rate both intact in the test split.
TEST_FRACTION = 0.2

SPLIT_OUTPUT_FILEPATHS = {
    'train': {
        'image':    os.path.join(TRAIN_REF_DIRPATH, 'fitzpatrick17k_train_image.txt'),
        'binary':   os.path.join(TRAIN_REF_DIRPATH, 'fitzpatrick17k_train_binary_label.txt'),
        'original': os.path.join(TRAIN_REF_DIRPATH, 'fitzpatrick17k_train_original_label.txt'),
        'scale':    os.path.join(TRAIN_REF_DIRPATH, 'fitzpatrick17k_train_fitzpatrick_scale.txt')
    },
    'test': {
        'image':    os.path.join(TEST_REF_DIRPATH, 'fitzpatrick17k_test_image.txt'),
        'binary':   os.path.join(TEST_REF_DIRPATH, 'fitzpatrick17k_test_binary_label.txt'),
        'original': os.path.join(TEST_REF_DIRPATH, 'fitzpatrick17k_test_original_label.txt'),
        'scale':    os.path.join(TEST_REF_DIRPATH, 'fitzpatrick17k_test_fitzpatrick_scale.txt')
    }
}

# Where the unsplit reference files live. Before the train/test split existed, every processed
# image was written straight to the training files, so that is what --splits_only reads back.
IMAGE_OUTPUT_FILEPATH = SPLIT_OUTPUT_FILEPATHS['train']['image']
BINARY_LABEL_OUTPUT_FILEPATH = SPLIT_OUTPUT_FILEPATHS['train']['binary']
ORIGINAL_LABEL_OUTPUT_FILEPATH = SPLIT_OUTPUT_FILEPATHS['train']['original']

def is_eczema_target(label):
    return eczema_labels.is_eczema_target('fitzpatrick17k', label)

def write_train_test_split(image_paths, binary_labels, original_labels, scales):
    '''
    Divides fitzpatrick17k into a training split and a skin-tone stratified test split

    Args:
        image_paths : list[str]
            all processed fitzpatrick17k image paths
        binary_labels : list[str]
            matching binary labels, '0' or '1'
        original_labels : list[str]
            matching free-text diagnosis labels
        scales : list[str]
            matching Fitzpatrick scale values, '1' to '6', or '-1' where unrated
    '''

    # Stratify on the pair so neither tone nor class drifts between splits
    strata = ['{}_{}'.format(scale, label) for scale, label in zip(scales, binary_labels)]

    train_indices, test_indices = data_utils.stratified_split(
        strata=strata,
        n_split=[1.0 - TEST_FRACTION, TEST_FRACTION])

    assert not (set(train_indices) & set(test_indices)), 'train and test splits overlap'
    assert sorted(train_indices + test_indices) == list(range(len(image_paths))), \
        'splits do not cover fitzpatrick17k'

    os.makedirs(TRAIN_REF_DIRPATH, exist_ok=True)
    os.makedirs(TEST_REF_DIRPATH, exist_ok=True)

    for split_name, indices in [('train', train_indices), ('test', test_indices)]:
        filepaths = SPLIT_OUTPUT_FILEPATHS[split_name]

        data_utils.write_paths(filepaths['image'], [image_paths[i] for i in indices])
        data_utils.write_paths(filepaths['binary'], [binary_labels[i] for i in indices])
        data_utils.write_paths(filepaths['original'], [original_labels[i] for i in indices])
        data_utils.write_paths(filepaths['scale'], [scales[i] for i in indices])

        n_positive = sum(1 for i in indices if binary_labels[i] == '1')
        by_tone = Counter(scales[i] for i in indices)

        print('Generated {} {} filepaths ({} eczema, {:.2%}) in {}'.format(
            len(indices), split_name, n_positive, n_positive / len(indices),
            os.path.dirname(filepaths['image'])))
        print('    Fitzpatrick type: {}'.format(
            '  '.join('{}={}'.format(tone, by_tone[tone]) for tone in sorted(by_tone, key=int))))

def read_scales_from_csv(root_dir):
    '''
    Recovers the Fitzpatrick scale for each already-processed image

    Images were written in CSV row order, skipping rows whose source file was missing, so
    replaying that filter reproduces the ordering of the existing reference files. The caller
    checks the recovered diagnosis labels against the stored ones before trusting the alignment.

    Args:
        root_dir : str
            Path to original raw dataset
    Returns:
        tuple : (list[str] diagnosis labels, list[str] Fitzpatrick scale values)
    '''

    csv_path = os.path.join(root_dir, 'fitzpatrick17k.csv')
    image_dir = os.path.join(root_dir, 'data', 'finalfitz17k')

    labels, scales = [], []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if not os.path.exists(os.path.join(image_dir, '{}.jpg'.format(row['md5hash']))):
                continue

            labels.append(row['label'])
            scales.append(row['fitzpatrick_scale'])

    return labels, scales

def setup_dataset_fitzpatrick17k(root_dir, output_dir):
    '''
    Sets up the Fitzpatrick17k directory. Images are stored in a flat directory
    (data/finalfitz17k/) named by their md5 hash, and labels come from the
    fitzpatrick17k.csv file.

    Args:
        root_dir : str
            Path to original raw dataset (contains data/finalfitz17k/ and fitzpatrick17k.csv)
        output_dir : str
            Path to save the derived dataset
    '''
    img_out_dir = os.path.join(output_dir, "images")

    csv_path = os.path.join(root_dir, "fitzpatrick17k.csv")
    image_dir = os.path.join(root_dir, "data", "finalfitz17k")

    if not os.path.exists(root_dir):
        print(f"[Error] Source directory not found: {root_dir}")
        return

    if not os.path.exists(csv_path):
        print(f"[Error] CSV file not found: {csv_path}")
        return

    if not os.path.exists(image_dir):
        print(f"[Error] Image directory not found: {image_dir}")
        return

    print(f"-> Processing Fitzpatrick17k from: {root_dir}")

    # Create necessary directories
    os.makedirs(img_out_dir, exist_ok=True)

    global_img_idx = 0

    train_image_paths = []
    train_binary_labels = []
    train_original_labels = []
    train_scales = []

    # Read CSV to build md5hash -> label mapping
    rows = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    skipped = 0

    for row in tqdm(rows, desc="Processing images"):
        md5hash = row['md5hash']
        label = row['label']

        # Find the corresponding image file
        img_filename = f"{md5hash}.jpg"
        orig_path = os.path.join(image_dir, img_filename)

        if not os.path.exists(orig_path):
            skipped += 1
            continue

        new_img_name = f"{global_img_idx}.png"
        image_path = os.path.join(img_out_dir, new_img_name)

        # Load the original image as a numpy array, ensure it is RGB, then pass to save_image
        try:
            img_np = np.array(Image.open(orig_path).convert('RGB'))
        except Exception as e:
            print(f"[Warning] Could not open image {orig_path}: {e}")
            skipped += 1
            continue

        data_utils.save_image(
            image=img_np,
            path=image_path,
            normalized=False,
            data_type='color',
            data_format='HWC'
        )

        binary_label = "1" if is_eczema_target(label) else "0"

        # Store relative paths with forward slashes for the dataloader
        rel_path = os.path.relpath(image_path).replace('\\', '/')
        train_image_paths.append(rel_path)
        train_binary_labels.append(binary_label)
        train_original_labels.append(label)
        train_scales.append(row['fitzpatrick_scale'])

        global_img_idx += 1

    if skipped > 0:
        print(f"[Info] Skipped {skipped} images (not found or unreadable)")

    write_train_test_split(
        train_image_paths, train_binary_labels, train_original_labels, train_scales)

if __name__ == "__main__":
    if args.splits_only:
        # The 16.5k image conversions above are unaffected by how the split is drawn.
        image_paths = data_utils.read_paths(IMAGE_OUTPUT_FILEPATH)
        binary_labels = data_utils.read_paths(BINARY_LABEL_OUTPUT_FILEPATH)
        original_labels = data_utils.read_paths(ORIGINAL_LABEL_OUTPUT_FILEPATH)

        csv_labels, scales = read_scales_from_csv(args.root_dir)

        # Skin tone is joined back on by position, so a drifted ordering must not pass silently
        assert csv_labels == original_labels, (
            'CSV order no longer matches {} ({} vs {} entries). Re-run the full setup instead '
            'of --splits_only.'.format(
                ORIGINAL_LABEL_OUTPUT_FILEPATH, len(csv_labels), len(original_labels)))

        write_train_test_split(image_paths, binary_labels, original_labels, scales)
    else:
        if args.output_dir is None:
            parser.error('--output_dir is required unless --splits_only is set')

        setup_dataset_fitzpatrick17k(root_dir=args.root_dir,
                                      output_dir=args.output_dir)
