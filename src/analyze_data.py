import os, sys
import numpy as np
from collections import Counter, OrderedDict

sys.path.insert(0, 'src')
import data_utils


'''
Data analysis script for the eczema binary classification datasets.

Reads the training and testing path files for each dataset (dermnet, fitzpatrick17k, sd198)
and prints useful statistics:
  - Total sample counts per split
  - Binary label distribution (eczema vs non-eczema) with class ratios
  - Original label (disease category) distribution and counts
  - Cross-dataset summary and combined statistics
'''

# Define all dataset configurations
DATASETS = OrderedDict({
    'dermnet': {
        'train': {
            'image': 'training/dermnet/dermnet_train_image.txt',
            'binary_label': 'training/dermnet/dermnet_train_binary_label.txt',
            'original_label': 'training/dermnet/dermnet_train_original_label.txt',
        },
        'test': {
            'image': 'testing/dermnet/dermnet_test_image.txt',
            'binary_label': 'testing/dermnet/dermnet_test_binary_label.txt',
            'original_label': 'testing/dermnet/dermnet_test_original_label.txt',
        },
    },
    'fitzpatrick17k': {
        'train': {
            'image': 'training/fitzpatrick17k/fitzpatrick17k_train_image.txt',
            'binary_label': 'training/fitzpatrick17k/fitzpatrick17k_train_binary_label.txt',
            'original_label': 'training/fitzpatrick17k/fitzpatrick17k_train_original_label.txt',
        },
    },
    'sd198': {
        'train': {
            'image': 'training/sd198/sd198_train_image.txt',
            'binary_label': 'training/sd198/sd198_train_binary_label.txt',
            'original_label': 'training/sd198/sd198_train_original_label.txt',
        },
    },
})


def print_separator(char='=', width=80):
    print(char * width)


def print_header(title, char='=', width=80):
    print()
    print_separator(char, width)
    print(f'  {title}')
    print_separator(char, width)


def analyze_binary_labels(binary_labels):
    '''
    Analyzes binary label distribution

    Args:
        binary_labels : list[str]
            list of '0' or '1' label strings
    Returns:
        dict : analysis results
    '''

    labels = [int(lbl) for lbl in binary_labels]
    counter = Counter(labels)

    n_total = len(labels)
    n_eczema = counter.get(1, 0)
    n_non_eczema = counter.get(0, 0)

    pct_eczema = (n_eczema / n_total * 100) if n_total > 0 else 0
    pct_non_eczema = (n_non_eczema / n_total * 100) if n_total > 0 else 0

    # Imbalance ratio (majority / minority)
    if n_eczema > 0 and n_non_eczema > 0:
        imbalance_ratio = max(n_eczema, n_non_eczema) / min(n_eczema, n_non_eczema)
    else:
        imbalance_ratio = float('inf')

    return {
        'n_total': n_total,
        'n_eczema': n_eczema,
        'n_non_eczema': n_non_eczema,
        'pct_eczema': pct_eczema,
        'pct_non_eczema': pct_non_eczema,
        'imbalance_ratio': imbalance_ratio,
    }


def analyze_original_labels(original_labels):
    '''
    Analyzes original label (disease category) distribution

    Args:
        original_labels : list[str]
            list of original disease category names
    Returns:
        dict : analysis results
    '''

    counter = Counter(original_labels)
    n_unique_classes = len(counter)

    # Sort by count (descending)
    sorted_classes = counter.most_common()

    counts = list(counter.values())
    mean_per_class = np.mean(counts)
    median_per_class = np.median(counts)
    min_per_class = np.min(counts)
    max_per_class = np.max(counts)
    std_per_class = np.std(counts)

    return {
        'n_unique_classes': n_unique_classes,
        'sorted_classes': sorted_classes,
        'mean_per_class': mean_per_class,
        'median_per_class': median_per_class,
        'min_per_class': min_per_class,
        'max_per_class': max_per_class,
        'std_per_class': std_per_class,
    }


def analyze_split(dataset_name, split_name, split_config):
    '''
    Analyzes a single dataset split (train or test)

    Args:
        dataset_name : str
            name of the dataset
        split_name : str
            'train' or 'test'
        split_config : dict
            paths to image, binary_label, and original_label files
    Returns:
        dict : analysis results for this split
    '''

    print_header(f'{dataset_name.upper()} — {split_name.upper()} Split', char='-')

    # Read paths and labels
    image_paths = data_utils.read_paths(split_config['image'])
    binary_labels = data_utils.read_paths(split_config['binary_label'])
    original_labels = data_utils.read_paths(split_config['original_label'])

    n_samples = len(image_paths)

    # Validate consistency
    assert n_samples == len(binary_labels), \
        f'Mismatch: {n_samples} images vs {len(binary_labels)} binary labels'
    assert n_samples == len(original_labels), \
        f'Mismatch: {n_samples} images vs {len(original_labels)} original labels'

    print(f'  Total samples: {n_samples}')
    print()

    # Binary label analysis
    binary_results = analyze_binary_labels(binary_labels)

    print(f'  Binary Label Distribution:')
    print(f'    Eczema (1):     {binary_results["n_eczema"]:>6}  ({binary_results["pct_eczema"]:.2f}%)')
    print(f'    Non-Eczema (0): {binary_results["n_non_eczema"]:>6}  ({binary_results["pct_non_eczema"]:.2f}%)')
    print(f'    Imbalance Ratio: {binary_results["imbalance_ratio"]:.2f}:1')
    print()

    # Original label analysis
    original_results = analyze_original_labels(original_labels)

    print(f'  Original Label (Disease Category) Statistics:')
    print(f'    Unique classes:    {original_results["n_unique_classes"]}')
    print(f'    Mean per class:    {original_results["mean_per_class"]:.1f}')
    print(f'    Median per class:  {original_results["median_per_class"]:.1f}')
    print(f'    Min per class:     {original_results["min_per_class"]}')
    print(f'    Max per class:     {original_results["max_per_class"]}')
    print(f'    Std per class:     {original_results["std_per_class"]:.1f}')
    print()

    # Print top 10 and bottom 5 classes
    top_n = min(10, len(original_results['sorted_classes']))
    print(f'  Top {top_n} classes by count:')
    for rank, (cls_name, count) in enumerate(original_results['sorted_classes'][:top_n], 1):
        pct = count / n_samples * 100
        print(f'    {rank:>3}. {cls_name:<55} {count:>5}  ({pct:.2f}%)')
    print()

    bottom_n = min(5, len(original_results['sorted_classes']))
    print(f'  Bottom {bottom_n} classes by count:')
    for cls_name, count in original_results['sorted_classes'][-bottom_n:]:
        pct = count / n_samples * 100
        print(f'       {cls_name:<55} {count:>5}  ({pct:.2f}%)')
    print()

    # Identify which original labels map to eczema
    eczema_classes = set()
    non_eczema_classes = set()

    for orig, binary in zip(original_labels, binary_labels):
        if int(binary) == 1:
            eczema_classes.add(orig)
        else:
            non_eczema_classes.add(orig)

    print(f'  Classes mapped to ECZEMA (label=1):')
    for cls_name in sorted(eczema_classes):
        count = Counter(original_labels)[cls_name]
        print(f'    - {cls_name} ({count} samples)')

    if len(eczema_classes) == 0:
        print(f'    (none)')
    print()

    return {
        'n_samples': n_samples,
        'binary_results': binary_results,
        'original_results': original_results,
        'eczema_classes': eczema_classes,
    }


def main():

    print_header('ECZEMA BINARY CLASSIFICATION — DATA ANALYSIS')

    all_results = {}

    # Analyze each dataset and split
    for dataset_name, splits in DATASETS.items():
        all_results[dataset_name] = {}

        for split_name, split_config in splits.items():
            # Check if files exist
            missing = [
                path for path in split_config.values()
                if not os.path.isfile(path)
            ]

            if missing:
                print(f'\n  [SKIP] {dataset_name}/{split_name} — missing files:')
                for m in missing:
                    print(f'    {m}')
                continue

            results = analyze_split(dataset_name, split_name, split_config)
            all_results[dataset_name][split_name] = results

    # Cross-dataset summary
    print_header('CROSS-DATASET SUMMARY')

    # Combined training statistics
    total_train_samples = 0
    total_train_eczema = 0
    total_train_non_eczema = 0
    all_train_eczema_classes = set()

    total_test_samples = 0
    total_test_eczema = 0
    total_test_non_eczema = 0

    print(f'\n  {"Dataset":<20} {"Split":<8} {"Total":>8} {"Eczema":>8} {"Non-Ecz":>8} {"Ecz %":>8} {"Imbalance":>10}')
    print(f'  {"-" * 20} {"-" * 8} {"-" * 8} {"-" * 8} {"-" * 8} {"-" * 8} {"-" * 10}')

    for dataset_name, splits in all_results.items():
        for split_name, results in splits.items():
            br = results['binary_results']
            print(f'  {dataset_name:<20} {split_name:<8} {br["n_total"]:>8} {br["n_eczema"]:>8} {br["n_non_eczema"]:>8} {br["pct_eczema"]:>7.2f}% {br["imbalance_ratio"]:>9.2f}:1')

            if split_name == 'train':
                total_train_samples += br['n_total']
                total_train_eczema += br['n_eczema']
                total_train_non_eczema += br['n_non_eczema']
                all_train_eczema_classes.update(results['eczema_classes'])

            elif split_name == 'test':
                total_test_samples += br['n_total']
                total_test_eczema += br['n_eczema']
                total_test_non_eczema += br['n_non_eczema']

    # Combined totals
    print(f'  {"-" * 20} {"-" * 8} {"-" * 8} {"-" * 8} {"-" * 8} {"-" * 8} {"-" * 10}')

    if total_train_samples > 0:
        train_pct = total_train_eczema / total_train_samples * 100
        train_imbalance = max(total_train_eczema, total_train_non_eczema) / max(min(total_train_eczema, total_train_non_eczema), 1)
        print(f'  {"COMBINED":<20} {"train":<8} {total_train_samples:>8} {total_train_eczema:>8} {total_train_non_eczema:>8} {train_pct:>7.2f}% {train_imbalance:>9.2f}:1')

    if total_test_samples > 0:
        test_pct = total_test_eczema / total_test_samples * 100
        test_imbalance = max(total_test_eczema, total_test_non_eczema) / max(min(total_test_eczema, total_test_non_eczema), 1)
        print(f'  {"COMBINED":<20} {"test":<8} {total_test_samples:>8} {total_test_eczema:>8} {total_test_non_eczema:>8} {test_pct:>7.2f}% {test_imbalance:>9.2f}:1')

    print()

    # Number of unique original classes across datasets
    print(f'  Number of unique classes per training dataset:')
    for dataset_name, splits in all_results.items():
        if 'train' in splits:
            n_classes = splits['train']['original_results']['n_unique_classes']
            print(f'    {dataset_name:<20} {n_classes} classes')
    print()

    # All eczema class names across datasets
    print(f'  All original class names mapped to eczema (label=1) across training sets:')
    for cls_name in sorted(all_train_eczema_classes):
        print(f'    - {cls_name}')
    print()

    # Recommendations for training
    print_header('TRAINING RECOMMENDATIONS')

    if total_train_samples > 0 and total_train_eczema > 0:
        combined_ratio = total_train_non_eczema / total_train_eczema
        print(f'  Combined class imbalance: {combined_ratio:.2f}:1 (non-eczema : eczema)')
        print()

        if combined_ratio > 5:
            weight_eczema = combined_ratio
            print(f'  [WARNING] Severe class imbalance detected.')
            print(f'  Recommended class weights for weighted cross entropy:')
            print(f'    --loss_func weighted_cross_entropy')
            print(f'    --loss_class_weights 1.0 {weight_eczema:.1f}')
            print()
        elif combined_ratio > 2:
            weight_eczema = combined_ratio
            print(f'  [NOTE] Moderate class imbalance detected.')
            print(f'  Consider using weighted cross entropy:')
            print(f'    --loss_func weighted_cross_entropy')
            print(f'    --loss_class_weights 1.0 {weight_eczema:.1f}')
            print()
        else:
            print(f'  Class distribution is relatively balanced. Standard cross entropy should work well.')
            print()

    print_separator()
    print('  Analysis complete.')
    print_separator()


if __name__ == '__main__':
    main()
