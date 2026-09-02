import os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, 'src')
import data_utils
import eczema_labels
import argparse
from tqdm import tqdm

# Parse arguments globally
parser = argparse.ArgumentParser(description="Process and split the Dermnet dataset for Eczema binary classification.")

parser.add_argument("--root_dir", type=str, help="Path to the raw Dermnet dataset directory (contains train/ and test/ subdirectories). Required unless --splits_only.")
parser.add_argument("--output_dir", type=str, help="Path to save the derived dataset and images. Required unless --splits_only.")
parser.add_argument("--splits_only", action="store_true", help="Rebuild the val/holdout split files from the existing reference files, skipping image processing.")

args = parser.parse_args()

TRAIN_REF_DIRPATH = os.path.join('training', 'dermnet')
TEST_REF_DIRPATH = os.path.join('testing', 'dermnet')
VAL_REF_DIRPATH = os.path.join('validation', 'dermnet')

TRAIN_IMAGE_OUTPUT_FILEPATH = os.path.join(TRAIN_REF_DIRPATH, 'dermnet_train_image.txt')
TRAIN_BINARY_LABEL_OUTPUT_FILEPATH = os.path.join(TRAIN_REF_DIRPATH, 'dermnet_train_binary_label.txt')
TRAIN_ORIGINAL_LABEL_OUTPUT_FILEPATH = os.path.join(TRAIN_REF_DIRPATH, 'dermnet_train_original_label.txt')

TEST_IMAGE_OUTPUT_FILEPATH = os.path.join(TEST_REF_DIRPATH, 'dermnet_test_image.txt')
TEST_BINARY_LABEL_OUTPUT_FILEPATH = os.path.join(TEST_REF_DIRPATH, 'dermnet_test_binary_label.txt')
TEST_ORIGINAL_LABEL_OUTPUT_FILEPATH = os.path.join(TEST_REF_DIRPATH, 'dermnet_test_original_label.txt')

# Dermnet's own test split is divided again, in half. The validation half drives checkpoint
# selection and threshold calibration during training; the holdout half is only read at reporting
# time. Splitting here rather than downstream keeps the two from ever being the same images.
VAL_FRACTION = 0.5

SPLIT_OUTPUT_FILEPATHS = {
    'val': {
        'image':    os.path.join(VAL_REF_DIRPATH, 'dermnet_val_image.txt'),
        'binary':   os.path.join(VAL_REF_DIRPATH, 'dermnet_val_binary_label.txt'),
        'original': os.path.join(VAL_REF_DIRPATH, 'dermnet_val_original_label.txt')
    },
    'holdout': {
        'image':    os.path.join(TEST_REF_DIRPATH, 'dermnet_holdout_image.txt'),
        'binary':   os.path.join(TEST_REF_DIRPATH, 'dermnet_holdout_binary_label.txt'),
        'original': os.path.join(TEST_REF_DIRPATH, 'dermnet_holdout_original_label.txt')
    }
}

def is_eczema_target(class_name):
    return eczema_labels.is_eczema_target('dermnet', class_name)

def process_split(split_dir, img_out_dir, global_img_idx):
    '''
    Processes one split (train or test) of the Dermnet dataset.

    Args:
        split_dir : str
            Path to the split directory (e.g. data/dermnet/train)
        img_out_dir : str
            Path to save flattened images
        global_img_idx : int
            Starting index for image naming
    Returns:
        tuple : (image_paths, binary_labels, original_labels, next_global_img_idx)
    '''

    image_paths = []
    binary_labels = []
    original_labels = []

    class_folders = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]

    for class_name in tqdm(class_folders, desc=f"Flattening classes in {os.path.basename(split_dir)}"):
        class_dir = os.path.join(split_dir, class_name)
        binary_label = "1" if is_eczema_target(class_name) else "0"

        # Grab valid images
        images = [img for img in os.listdir(class_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for img_name in images:
            new_img_name = f"{global_img_idx}.png"
            image_path = os.path.join(img_out_dir, new_img_name)

            # Load the original image as a numpy array, ensure it is RGB, then pass to save_image
            orig_path = os.path.join(class_dir, img_name)
            img_np = np.array(Image.open(orig_path).convert('RGB'))

            data_utils.save_image(
                image=img_np,
                path=image_path,
                normalized=False,
                data_type='color',
                data_format='HWC'
            )

            # Store relative paths with forward slashes for the dataloader
            rel_path = os.path.relpath(image_path).replace('\\', '/')
            image_paths.append(rel_path)
            binary_labels.append(binary_label)
            original_labels.append(class_name)

            global_img_idx += 1

    return image_paths, binary_labels, original_labels, global_img_idx

def write_val_holdout_split(image_paths, binary_labels, original_labels):
    '''
    Divides the dermnet test split into a validation half and a reporting half

    Stratified on the binary label so both halves carry the same eczema rate.

    Args:
        image_paths : list[str]
            image paths of the full dermnet test split
        binary_labels : list[str]
            matching binary labels, '0' or '1'
        original_labels : list[str]
            matching original dermnet class names, kept so false positives can later be
            attributed to a diagnosis
    '''

    val_indices, holdout_indices = data_utils.stratified_split(
        strata=binary_labels,
        n_split=[VAL_FRACTION, 1.0 - VAL_FRACTION])

    assert not (set(val_indices) & set(holdout_indices)), 'val and holdout splits overlap'
    assert sorted(val_indices + holdout_indices) == list(range(len(image_paths))), \
        'splits do not cover the dermnet test set'

    os.makedirs(VAL_REF_DIRPATH, exist_ok=True)
    os.makedirs(TEST_REF_DIRPATH, exist_ok=True)

    for split_name, indices in [('val', val_indices), ('holdout', holdout_indices)]:
        filepaths = SPLIT_OUTPUT_FILEPATHS[split_name]

        data_utils.write_paths(filepaths['image'], [image_paths[i] for i in indices])
        data_utils.write_paths(filepaths['binary'], [binary_labels[i] for i in indices])
        data_utils.write_paths(filepaths['original'], [original_labels[i] for i in indices])

        n_positive = sum(1 for i in indices if binary_labels[i] == '1')
        print('Generated {} {} filepaths ({} eczema, {:.2%}) in {}'.format(
            len(indices), split_name, n_positive, n_positive / len(indices),
            os.path.dirname(filepaths['image'])))

def setup_dataset_dermnet(root_dir, output_dir):
    '''
    Sets up the Dermnet directory. Dermnet has separate train/ and test/ subdirectories,
    each containing class-named folders with images.

    Args:
        root_dir : str
            Path to original raw dataset (contains train/ and test/ subdirs)
        output_dir : str
            Path to save the derived dataset
    '''
    img_out_dir = os.path.join(output_dir, "images")

    train_dir = os.path.join(root_dir, "train")
    test_dir = os.path.join(root_dir, "test")

    if not os.path.exists(root_dir):
        print(f"[Error] Source directory not found: {root_dir}")
        return

    if not os.path.exists(train_dir):
        print(f"[Error] Train directory not found: {train_dir}")
        return

    if not os.path.exists(test_dir):
        print(f"[Error] Test directory not found: {test_dir}")
        return

    print(f"-> Processing Dermnet from: {root_dir}")

    # Create necessary directories
    os.makedirs(img_out_dir, exist_ok=True)

    global_img_idx = 0

    # Process train split
    print("Processing train split...")
    train_image_paths, train_binary_labels, train_original_labels, global_img_idx = \
        process_split(train_dir, img_out_dir, global_img_idx)

    # Process test split
    print("Processing test split...")
    test_image_paths, test_binary_labels, test_original_labels, global_img_idx = \
        process_split(test_dir, img_out_dir, global_img_idx)

    # Write Train Outputs
    os.makedirs(TRAIN_REF_DIRPATH, exist_ok=True)

    data_utils.write_paths(TRAIN_IMAGE_OUTPUT_FILEPATH, train_image_paths)
    data_utils.write_paths(TRAIN_BINARY_LABEL_OUTPUT_FILEPATH, train_binary_labels)
    data_utils.write_paths(TRAIN_ORIGINAL_LABEL_OUTPUT_FILEPATH, train_original_labels)

    print(f"Generated {len(train_image_paths)} training filepaths in {TRAIN_REF_DIRPATH}")

    # Write Test Outputs
    os.makedirs(TEST_REF_DIRPATH, exist_ok=True)

    data_utils.write_paths(TEST_IMAGE_OUTPUT_FILEPATH, test_image_paths)
    data_utils.write_paths(TEST_BINARY_LABEL_OUTPUT_FILEPATH, test_binary_labels)
    data_utils.write_paths(TEST_ORIGINAL_LABEL_OUTPUT_FILEPATH, test_original_labels)

    print(f"Generated {len(test_image_paths)} testing filepaths in {TEST_REF_DIRPATH}")

    write_val_holdout_split(test_image_paths, test_binary_labels, test_original_labels)

if __name__ == "__main__":
    if args.splits_only:
        # Re-deriving the split does not need any of the 15k image conversions above.
        write_val_holdout_split(
            data_utils.read_paths(TEST_IMAGE_OUTPUT_FILEPATH),
            data_utils.read_paths(TEST_BINARY_LABEL_OUTPUT_FILEPATH),
            data_utils.read_paths(TEST_ORIGINAL_LABEL_OUTPUT_FILEPATH))
    else:
        if args.root_dir is None or args.output_dir is None:
            parser.error('--root_dir and --output_dir are required unless --splits_only is set')

        setup_dataset_dermnet(root_dir=args.root_dir,
                              output_dir=args.output_dir)
