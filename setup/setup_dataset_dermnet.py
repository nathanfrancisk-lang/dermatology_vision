import os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, 'src')
import data_utils
import argparse
from tqdm import tqdm

# Parse arguments globally
parser = argparse.ArgumentParser(description="Process and split the Dermnet dataset for Eczema binary classification.")

parser.add_argument("--root_dir", type=str, required=True, help="Path to the raw Dermnet dataset directory (contains train/ and test/ subdirectories).")
parser.add_argument("--output_dir", type=str, required=True, help="Path to save the derived dataset and images.")

args = parser.parse_args()

TRAIN_REF_DIRPATH = os.path.join('training', 'dermnet')
TEST_REF_DIRPATH = os.path.join('testing', 'dermnet')

TRAIN_IMAGE_OUTPUT_FILEPATH = os.path.join(TRAIN_REF_DIRPATH, 'dermnet_train_image.txt')
TRAIN_BINARY_LABEL_OUTPUT_FILEPATH = os.path.join(TRAIN_REF_DIRPATH, 'dermnet_train_binary_label.txt')
TRAIN_ORIGINAL_LABEL_OUTPUT_FILEPATH = os.path.join(TRAIN_REF_DIRPATH, 'dermnet_train_original_label.txt')

TEST_IMAGE_OUTPUT_FILEPATH = os.path.join(TEST_REF_DIRPATH, 'dermnet_test_image.txt')
TEST_BINARY_LABEL_OUTPUT_FILEPATH = os.path.join(TEST_REF_DIRPATH, 'dermnet_test_binary_label.txt')
TEST_ORIGINAL_LABEL_OUTPUT_FILEPATH = os.path.join(TEST_REF_DIRPATH, 'dermnet_test_original_label.txt')

def is_eczema_target(class_name):
    keywords = ["eczema", "atopic dermatitis"]
    class_lower = class_name.lower()
    return any(keyword in class_lower for keyword in keywords)

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

            # Store absolute paths for the dataloader
            image_paths.append(os.path.abspath(image_path))
            binary_labels.append(binary_label)
            original_labels.append(class_name)

            global_img_idx += 1

    return image_paths, binary_labels, original_labels, global_img_idx

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

if __name__ == "__main__":
    setup_dataset_dermnet(root_dir=args.root_dir,
                          output_dir=args.output_dir)
