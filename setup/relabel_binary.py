'''
Regenerates the binary eczema label files from the existing original label files.

The setup_dataset_*.py scripts flatten and copy ~42k images; changing only the eczema class
definition does not require redoing any of that. This rewrites the *_binary_label.txt files
in place from *_original_label.txt using the current eczema_labels definitions, and reports
what moved.

Usage:  python setup/relabel_binary.py [--dry_run]
'''

import os, sys
import argparse
from collections import Counter

sys.path.insert(0, 'src')
import data_utils
import eczema_labels

# (dataset, split, reference directory)
SPLITS = [
    ('dermnet', 'train', os.path.join('training', 'dermnet')),
    ('dermnet', 'test', os.path.join('testing', 'dermnet')),
    ('fitzpatrick17k', 'train', os.path.join('training', 'fitzpatrick17k')),
    ('sd198', 'train', os.path.join('training', 'sd198')),
]


def relabel_split(dataset, split, ref_dirpath, dry_run=False):
    '''
    Rewrites one split's binary label file from its original label file

    Args:
        dataset : str
            One of dermnet, fitzpatrick17k, sd198
        split : str
            train or test
        ref_dirpath : str
            Directory holding the reference txt files
        dry_run : bool
            if set, report changes without writing
    Returns:
        dict : summary counts for this split
    '''

    original_filepath = os.path.join(ref_dirpath, '{}_{}_original_label.txt'.format(dataset, split))
    binary_filepath = os.path.join(ref_dirpath, '{}_{}_binary_label.txt'.format(dataset, split))

    original_labels = data_utils.read_paths(original_filepath)
    old_binary_labels = data_utils.read_paths(binary_filepath)

    assert len(original_labels) == len(old_binary_labels), \
        '{}: {} original labels but {} binary labels'.format(
            binary_filepath, len(original_labels), len(old_binary_labels))

    new_binary_labels = [
        '1' if eczema_labels.is_eczema_target(dataset, label) else '0'
        for label in original_labels
    ]

    n_positive = new_binary_labels.count('1')

    # A class-name typo would silently label everything negative, which trains without error
    assert n_positive > 0, \
        '{} {}: no positives matched. Class names in eczema_labels do not match this dataset.'.format(
            dataset, split)

    # Which original classes changed side
    promoted = Counter()
    demoted = Counter()
    for original, old, new in zip(original_labels, old_binary_labels, new_binary_labels):
        if old == '0' and new == '1':
            promoted[original] += 1
        elif old == '1' and new == '0':
            demoted[original] += 1

    if not dry_run:
        data_utils.write_paths(binary_filepath, new_binary_labels)

    return {
        'dataset': dataset,
        'split': split,
        'n_total': len(new_binary_labels),
        'n_positive_old': old_binary_labels.count('1'),
        'n_positive_new': n_positive,
        'promoted': promoted,
        'demoted': demoted,
    }


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Regenerate binary eczema labels from original labels.')
    parser.add_argument('--dry_run',
        action='store_true', help='Report changes without writing files')
    args = parser.parse_args()

    summaries = [relabel_split(*split, dry_run=args.dry_run) for split in SPLITS]

    for summary in summaries:
        print('{} {}  n={}  positives {} -> {}'.format(
            summary['dataset'],
            summary['split'],
            summary['n_total'],
            summary['n_positive_old'],
            summary['n_positive_new']))

        for class_name, count in summary['promoted'].most_common():
            print('    + {:5d}  {}'.format(count, class_name))
        for class_name, count in summary['demoted'].most_common():
            print('    - {:5d}  {}'.format(count, class_name))

    n_train_total = sum(s['n_total'] for s in summaries if s['split'] == 'train')
    n_train_positive = sum(s['n_positive_new'] for s in summaries if s['split'] == 'train')

    print('')
    print('Train positive rate: {}/{} = {:.1%}'.format(
        n_train_positive, n_train_total, n_train_positive / n_train_total))

    if args.dry_run:
        print('Dry run, no files written.')
