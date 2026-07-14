import os, sys
import csv
import numpy as np
from PIL import Image
sys.path.insert(0, 'src')
import data_utils
import argparse
from tqdm import tqdm

# Parse arguments globally
parser = argparse.ArgumentParser(description="Process the Fitzpatrick17k dataset for Eczema binary classification.")

parser.add_argument("--root_dir", type=str, required=True, help="Path to the raw Fitzpatrick17k dataset directory (contains data/finalfitz17k/ and fitzpatrick17k.csv).")
parser.add_argument("--output_dir", type=str, required=True, help="Path to save the derived dataset and images.")

args = parser.parse_args()

TRAIN_REF_DIRPATH = os.path.join('training', 'fitzpatrick17k')

IMAGE_OUTPUT_FILEPATH = os.path.join(TRAIN_REF_DIRPATH, 'fitzpatrick17k_train_image.txt')
BINARY_LABEL_OUTPUT_FILEPATH = os.path.join(TRAIN_REF_DIRPATH, 'fitzpatrick17k_train_binary_label.txt')
ORIGINAL_LABEL_OUTPUT_FILEPATH = os.path.join(TRAIN_REF_DIRPATH, 'fitzpatrick17k_train_original_label.txt')

def is_eczema_target(label):
    keywords = ["eczema", "atopic dermatitis"]
    label_lower = label.lower()
    return any(keyword in label_lower for keyword in keywords)

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

        global_img_idx += 1

    if skipped > 0:
        print(f"[Info] Skipped {skipped} images (not found or unreadable)")

    # Write Outputs
    os.makedirs(TRAIN_REF_DIRPATH, exist_ok=True)

    data_utils.write_paths(IMAGE_OUTPUT_FILEPATH, train_image_paths)
    data_utils.write_paths(BINARY_LABEL_OUTPUT_FILEPATH, train_binary_labels)
    data_utils.write_paths(ORIGINAL_LABEL_OUTPUT_FILEPATH, train_original_labels)

    print(f"Generated {len(train_image_paths)} training filepaths in {TRAIN_REF_DIRPATH}")

if __name__ == "__main__":
    setup_dataset_fitzpatrick17k(root_dir=args.root_dir,
                                  output_dir=args.output_dir)
