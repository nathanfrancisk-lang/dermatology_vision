'''
Self-checks for the pieces of the pipeline that fail silently.

Everything here guards against a bug that produces no error and no crash, only worse numbers:
augmentation that is declared but never applied, evaluation that is accidentally
non-deterministic, metrics that report the wrong class, or a label mapping whose class names
have drifted away from the dataset.

Run:  python src/test_pipeline.py
'''

import os, sys
import numpy as np

sys.path.insert(0, 'src')

import datasets
import eval_utils
import eczema_labels


def test_eczema_labels():
    '''Class names in the mapping must still match the datasets they refer to.'''

    # Traps the old keyword match got wrong, in both directions
    assert eczema_labels.is_eczema_target('sd198', 'Infantile_Atopic_Dermatitis')
    assert eczema_labels.is_eczema_target('sd198', 'Erythema_Craquele')
    assert eczema_labels.is_eczema_target('dermnet', 'Poison Ivy Photos and other Contact Dermatitis')

    assert not eczema_labels.is_eczema_target('sd198', 'Perioral_Dermatitis')
    assert not eczema_labels.is_eczema_target('sd198', 'Factitial_Dermatitis')
    assert not eczema_labels.is_eczema_target('sd198', 'Seborrheic_Keratosis')
    assert not eczema_labels.is_eczema_target('sd198', 'Xerosis')
    assert not eczema_labels.is_eczema_target('sd198', 'Cutaneous_T-Cell_Lymphoma')
    assert not eczema_labels.is_eczema_target('fitzpatrick17k', 'mycosis fungoides')
    assert not eczema_labels.is_eczema_target('fitzpatrick17k', 'psoriasis')

    # fitzpatrick17k labels come from free text, so matching must survive case and whitespace
    assert eczema_labels.is_eczema_target('fitzpatrick17k', '  Allergic Contact Dermatitis ')

    print('  labels: mapping traps hold')


def test_label_files_match_mapping():
    '''
    Every positive class in the mapping must appear in the dataset it belongs to.

    A renamed class would silently drop its images to negative, which trains without error.
    Skipped when the label files are not present.
    '''

    splits = [
        ('dermnet', 'training/dermnet/dermnet_train_original_label.txt'),
        ('fitzpatrick17k', 'training/fitzpatrick17k/fitzpatrick17k_train_original_label.txt'),
        ('sd198', 'training/sd198/sd198_train_original_label.txt'),
    ]

    for dataset, filepath in splits:
        if not os.path.exists(filepath):
            print('  labels: {} not set up, skipped'.format(dataset))
            continue

        with open(filepath) as f:
            present = set(line.strip() for line in f if line.strip())

        if dataset == 'fitzpatrick17k':
            present = set(name.lower() for name in present)

        missing = eczema_labels.ECZEMA_CLASSES[dataset] - present

        assert not missing, \
            '{}: eczema classes not found in dataset, names have drifted: {}'.format(
                dataset, sorted(missing))

        print('  labels: all {} eczema classes present in {}'.format(
            len(eczema_labels.ECZEMA_CLASSES[dataset]), dataset))


def test_augmentation_applies_to_train_only():
    '''
    Train augmentation must actually change pixels, and eval must not.

    The augmentation arguments were previously parsed, threaded through train() and
    documented while never being applied to a single image. Nothing failed; the model just
    memorized the training set. This is the check that would have caught it.
    '''

    image = np.random.randint(0, 256, size=(3, 64, 64)).astype(np.float32)

    augmentation = {
        'random_brightness': [0.8, 1.2],
        'random_contrast': [0.8, 1.2],
        'random_saturation': [0.8, 1.2],
        'random_hue': [-0.1, 0.1],
        'random_flip_type': ['horizontal', 'vertical'],
        'random_rotate_max': 15,
    }

    train_dataset = datasets.EczemaDataset(
        dataset='train', image_paths=['x'], labels=[1], shape=None, **augmentation)

    # Same augmentation config, eval split: must be ignored entirely
    val_dataset = datasets.EczemaDataset(
        dataset='val', image_paths=['x'], labels=[1], shape=None, **augmentation)

    train_draws = [train_dataset._augment(image) for _ in range(8)]

    assert any(not np.array_equal(draw, image) for draw in train_draws), \
        'train augmentation never changed the image: the transforms are not being applied'

    assert any(not np.array_equal(train_draws[0], draw) for draw in train_draws[1:]), \
        'train augmentation produced the same result every time: it is not random'

    assert not val_dataset.do_augment, \
        'validation split has augmentation enabled; evaluation must be deterministic'

    print('  augment: train transforms apply and vary, val is deterministic')


def test_metrics_report_the_positive_class():
    '''
    Reported sensitivity must be the eczema class, not the macro average.

    Built to mimic the real imbalance: on a ~90% negative set the macro recall reads far
    higher than eczema recall, and the two were being confused.
    '''

    labels = np.array([1] * 100 + [0] * 900)

    # Catch half the eczema cases, almost no false positives
    predictions = np.array([1] * 50 + [0] * 50 + [0] * 890 + [1] * 10)
    probabilities = predictions.astype(np.float64)

    results = eval_utils.evaluate(predictions, probabilities, labels)

    sensitivity = results['recall_per_class'][eval_utils.POSITIVE_CLASS]

    assert np.isclose(sensitivity, 0.5), \
        'expected eczema sensitivity 0.5, got {}'.format(sensitivity)

    assert results['recall_macro'] > 0.7, \
        'test setup is wrong: macro recall should be inflated relative to sensitivity'

    lines = '\n'.join(eval_utils.format_results(results))

    assert 'Sensitivity (eczema):  0.50000' in lines, \
        'format_results is not reporting positive-class sensitivity:\n{}'.format(lines)

    print('  metrics: sensitivity reports the eczema class ({:.2f} vs macro {:.2f})'.format(
        sensitivity, results['recall_macro']))


def test_specificity_at_sensitivity():
    '''The operating-point helper must hit the requested sensitivity, not merely approach it.'''

    labels = np.array([1] * 100 + [0] * 900)

    # Positives score higher on average but the classes overlap
    rng = np.random.default_rng(13)
    probabilities = np.concatenate([
        rng.uniform(0.3, 1.0, size=100),
        rng.uniform(0.0, 0.7, size=900),
    ])

    for target in [0.8, 0.9, 0.95]:
        specificity, threshold = eval_utils.specificity_at_sensitivity(
            probabilities, labels, target)

        achieved = (probabilities[labels == 1] >= threshold).mean()

        assert achieved >= target, \
            'target {} but threshold {:.4g} only achieves sensitivity {:.4f}'.format(
                target, threshold, achieved)

        assert 0.0 <= specificity <= 1.0

    # Higher sensitivity must never cost less specificity
    specificity_80, _ = eval_utils.specificity_at_sensitivity(probabilities, labels, 0.8)
    specificity_95, _ = eval_utils.specificity_at_sensitivity(probabilities, labels, 0.95)

    assert specificity_80 >= specificity_95, \
        'specificity should be non-increasing in target sensitivity'

    # Unreachable target must not silently return a plausible-looking operating point
    assert eval_utils.specificity_at_sensitivity(probabilities, labels, 1.5) == (0.0, 0.0)

    print('  threshold: operating points reach their target sensitivity')


if __name__ == '__main__':

    print('Running pipeline self-checks...')

    test_eczema_labels()
    test_label_files_match_mapping()
    test_augmentation_applies_to_train_only()
    test_metrics_report_the_positive_class()
    test_specificity_at_sensitivity()

    print('All checks passed.')
