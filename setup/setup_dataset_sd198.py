import os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, 'src')
import data_utils
import argparse
from tqdm import tqdm

# Parse arguments globally
parser = argparse.ArgumentParser(description="Process and split the SD-198 dataset for Eczema binary classification.")

parser.add_argument("--root_dir", type=str, required=True, help="Path to the raw SD-198 dataset directory.")
parser.add_argument("--output_dir", type=str, required=True, help="Path to save the derived dataset and images.")

args = parser.parse_args()

TRAIN_REF_DIRPATH = os.path.join('training', 'sd198')

IMAGE_OUTPUT_FILEPATH = os.path.join(TRAIN_REF_DIRPATH, 'sd198_train_image.txt')
BINARY_LABEL_OUTPUT_FILEPATH = os.path.join(TRAIN_REF_DIRPATH, 'sd198_train_binary_label.txt')
ORIGINAL_LABEL_OUTPUT_FILEPATH = os.path.join(TRAIN_REF_DIRPATH, 'sd198_train_original_label.txt')

def is_eczema_target(class_name):
    keywords = ["eczema", "atopic dermatitis"]
    class_lower = class_name.lower()
    return any(keyword in class_lower for keyword in keywords)

def setup_dataset_sd198(root_dir, output_dir):
    '''
    Sets up the SD198 directory

    Args:
        rood_dir : str
            Path to original raw dataset
        output_dir : str
            Path to save the derived dataset
    '''
    img_out_dir = os.path.join(output_dir, "images")

    if not os.path.exists(root_dir):
        print(f"[Error] Source directory not found: {root_dir}")
        return

    print(f"-> Processing SD-198 from: {root_dir}")

    # Create necessary directories
    os.makedirs(img_out_dir, exist_ok=True)

    global_img_idx = 0

    train_image_paths = []
    train_binary_labels = []
    train_original_labels = []

    class_folders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    for class_name in tqdm(class_folders, desc="Flattening classes"):
        class_dir = os.path.join(root_dir, class_name)
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
            train_image_paths.append(rel_path)
            train_binary_labels.append(binary_label)
            train_original_labels.append(class_name)

            global_img_idx += 1

    # Write Outputs
    os.makedirs(TRAIN_REF_DIRPATH, exist_ok=True)

    data_utils.write_paths(IMAGE_OUTPUT_FILEPATH, train_image_paths)
    data_utils.write_paths(BINARY_LABEL_OUTPUT_FILEPATH, train_binary_labels)
    data_utils.write_paths(ORIGINAL_LABEL_OUTPUT_FILEPATH, train_original_labels)

    print(f"Generated {len(train_image_paths)} training filepaths in {TRAIN_REF_DIRPATH}")

if __name__ == "__main__":
    setup_dataset_sd198(root_dir=args.root_dir,
                        output_dir=args.output_dir)