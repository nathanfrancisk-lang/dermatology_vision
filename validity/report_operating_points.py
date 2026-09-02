'''
Held-out performance report for the eczema binary classifier.

Organised around what the model is actually good at. At a 12% eczema base rate it rules eczema
OUT far more reliably than it rules it in, so the rule-out operating point leads and the
sensitivity-target points follow as secondary detail.

Every threshold is chosen on the validation split and applied unchanged to the held-out split,
so no reported number is tuned on the data it is reported over. Reads the probabilities.txt that
run_eczema_binary_classification.py writes with --save_outputs.

Run from the repository root:  python validity/report_operating_points.py --help
'''
import argparse, os, sys
import numpy as np
from sklearn.metrics import roc_curve

sys.path.insert(0, 'src')
import eval_utils


# Dermnet's two malignant classes. Flagging one of these as eczema routes a cancer to a topical
# steroid, which is the one false positive that is not recoverable at the next visit.
DEFAULT_MALIGNANT_LABELS = [
    'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
    'Melanoma Skin Cancer Nevi and Moles'
]

parser = argparse.ArgumentParser()
parser.add_argument('--val_probabilities_file',
    type=str, required=True, help='probabilities.txt from inference on the validation split')
parser.add_argument('--val_labels_file',
    type=str, required=True, help='Ground truth labels for the validation split')
parser.add_argument('--test_probabilities_file',
    type=str, required=True, help='probabilities.txt from inference on the held-out split')
parser.add_argument('--test_labels_file',
    type=str, required=True, help='Ground truth labels for the held-out split')
parser.add_argument('--val_original_labels_file',
    type=str, default=None, help='Original diagnosis names for the validation split; required to calibrate the malignancy-constrained threshold')
parser.add_argument('--test_original_labels_file',
    type=str, default=None, help='Original diagnosis names for the held-out split; enables the false positive and malignancy breakdowns')
parser.add_argument('--test_scales_file',
    type=str, default=None, help='Fitzpatrick scale values for the held-out split; enables the skin tone breakdown')
parser.add_argument('--target_npv',
    type=float, default=0.98, help='Negative predictive value the rule-out zone must hold')
parser.add_argument('--max_malignant_leakage',
    type=float, default=0.05, help='Largest fraction of malignant lesions allowed to be flagged eczema')
parser.add_argument('--malignant_labels',
    nargs='+', type=str, default=DEFAULT_MALIGNANT_LABELS, help='Original label names counted as malignant')
parser.add_argument('--target_sensitivities',
    nargs='+', type=float, default=[0.85, 0.90, 0.95], help='Sensitivity targets for the secondary table')
parser.add_argument('--n_bootstrap',
    type=int, default=2000, help='Bootstrap resamples for confidence intervals; 0 disables')
parser.add_argument('--output_path',
    type=str, default=None, help='Directory to write the report, roc.csv and roc.png to')


def load(probabilities_file, labels_file):
    probabilities = np.loadtxt(probabilities_file, dtype=np.float32, ndmin=1)
    labels = np.loadtxt(labels_file, dtype=np.int64, ndmin=1)
    assert len(probabilities) == len(labels), '{} probabilities vs {} labels in {}'.format(
        len(probabilities), len(labels), probabilities_file)
    return probabilities, labels

def read_column(filepath, n_expect):
    values = np.array([line.strip() for line in open(filepath) if line.strip()])
    assert len(values) == n_expect, \
        '{} holds {} entries, expected {}'.format(filepath, len(values), n_expect)
    return values

def bootstrap_interval(probabilities, labels, statistic, n_bootstrap, seed=13):
    '''
    Percentile bootstrap confidence interval for a metric over a fixed evaluation set

    Answers how much of a reported number is the model and how much is the particular sample it
    was measured on. Resampling is over cases, which is the sampling variation that a different
    test set of the same size would show.

    Args:
        probabilities : numpy[float]
            predicted probabilities for the positive class
        labels : numpy[int]
            ground truth class labels
        statistic : callable
            maps (probabilities, labels) to a float
        n_bootstrap : int
            number of resamples
        seed : int
            seed for the resampling
    Returns:
        tuple : (2.5th percentile, 97.5th percentile), or (nan, nan) if n_bootstrap is 0
    '''

    if n_bootstrap <= 0:
        return float('nan'), float('nan')

    rng = np.random.default_rng(seed)
    values = []

    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(labels), len(labels))

        # A resample that lost one of the classes cannot produce a sensitivity or an AUC
        if len(np.unique(labels[indices])) < 2:
            continue

        values.append(statistic(probabilities[indices], labels[indices]))

    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))

def rule_out_threshold(probabilities, labels, target_npv):
    '''
    Highest cut below which the model can declare "not eczema" and still hold the target NPV

    Negative predictive value is the share of cases below the cut that really are not eczema.
    Raising the cut clears more cases but eventually starts sweeping real eczema in with them,
    so the largest cut meeting the target is the most useful one.

    Args:
        probabilities : numpy[float]
            predicted probabilities for the positive class
        labels : numpy[int]
            ground truth class labels
        target_npv : float
            negative predictive value the zone must hold
    Returns:
        float : threshold, or 0.0 if no cut reaches the target
    '''

    best = 0.0

    for candidate in np.unique(probabilities):
        zone = probabilities <= candidate

        if zone.sum() == 0:
            continue

        npv = (labels[zone] == 0).mean()

        if npv >= target_npv:
            best = float(candidate)

    return best

def leakage_threshold(probabilities, is_malignant, max_leakage):
    '''
    Lowest cut at which no more than max_leakage of malignant lesions are called eczema

    Lowest rather than highest because the cut should be as permissive as the safety budget
    allows: every step up costs sensitivity, so the constraint is spent, not exceeded.

    Args:
        probabilities : numpy[float]
            predicted probabilities for the positive class
        is_malignant : numpy[bool]
            whether each case is a malignant lesion
        max_leakage : float
            largest acceptable fraction of malignant lesions flagged eczema
    Returns:
        float : threshold, or 1.0 if the budget is unreachable
    '''

    if is_malignant.sum() == 0:
        return 1.0

    for candidate in np.unique(probabilities):
        if (probabilities[is_malignant] >= candidate).mean() <= max_leakage:
            return float(candidate)

    return 1.0

def point_metrics(probabilities, labels, threshold):
    '''
    Metrics for one threshold, including the predictive values that depend on base rate

    Args:
        probabilities : numpy[float]
            predicted probabilities for the positive class
        labels : numpy[int]
            ground truth class labels
        threshold : float
            decision threshold
    Returns:
        dict : eval_utils.evaluate output plus sensitivity, specificity, ppv and npv
    '''

    predictions = (probabilities >= threshold).astype(np.int64)
    results = eval_utils.evaluate(predictions, probabilities, labels)

    results['sensitivity'] = results['recall_per_class'][eval_utils.POSITIVE_CLASS]
    results['ppv'] = results['precision_per_class'][eval_utils.POSITIVE_CLASS]

    n_negative_call = results['tn'] + results['fn']
    results['npv'] = results['tn'] / n_negative_call if n_negative_call > 0 else float('nan')

    return results


if __name__ == '__main__':
    args = parser.parse_args()

    val_probabilities, val_labels = load(args.val_probabilities_file, args.val_labels_file)
    test_probabilities, test_labels = load(args.test_probabilities_file, args.test_labels_file)

    lines = []
    log = lambda line: (print(line), lines.append(line))

    prevalence = test_labels.mean()

    log('Held-out set: {} images, {} eczema ({:.2%} prevalence)'.format(
        len(test_labels), int(test_labels.sum()), prevalence))
    log('')

    '''
    Rule-out: the primary claim
    '''
    threshold = rule_out_threshold(val_probabilities, val_labels, args.target_npv)
    zone = test_probabilities <= threshold

    log('=' * 78)
    log('RULE-OUT  (primary)  -  "this is not eczema", cleared without specialist review')
    log('=' * 78)

    if zone.sum() == 0:
        log('No rule-out zone reaches NPV {:.1%} on validation.'.format(args.target_npv))
    else:
        npv = float((test_labels[zone] == 0).mean())
        npv_low, npv_high = bootstrap_interval(
            test_probabilities, test_labels,
            lambda p, l, t=threshold: (l[p <= t] == 0).mean() if (p <= t).sum() > 0 else np.nan,
            args.n_bootstrap)

        log('  Threshold          p <= {:.4g}   (calibrated on validation for NPV {:.0%})'.format(
            threshold, args.target_npv))
        log('  Coverage           {}/{} = {:.1%} of all cases cleared'.format(
            int(zone.sum()), len(test_labels), zone.mean()))
        log('  NPV                {:.4f}   95% CI [{:.4f}, {:.4f}]'.format(npv, npv_low, npv_high))
        log('  Eczema missed      {} of {} ({:.1%} of the cleared cases)'.format(
            int(test_labels[zone].sum()), int(test_labels.sum()),
            test_labels[zone].mean()))
        log('')
        log('  Baseline for comparison: clearing nothing leaves the {:.1%} base rate of eczema'.format(
            prevalence))
        log('  among unreviewed cases. This zone cuts that to {:.1%}, a {:.1f}x reduction.'.format(
            test_labels[zone].mean(), prevalence / max(test_labels[zone].mean(), 1e-9)))
        log('')
        log('  NPV depends on prevalence. At this threshold the model holds sensitivity {:.3f}'.format(
            point_metrics(test_probabilities, test_labels, threshold)['sensitivity']))
        log('  and specificity {:.3f}, which are prevalence-independent; NPV at other base rates:'.format(
            point_metrics(test_probabilities, test_labels, threshold)['specificity']))

        metrics = point_metrics(test_probabilities, test_labels, threshold)
        miss_rate = 1.0 - metrics['sensitivity']
        for other in [0.10, 0.20, 0.30, 0.50]:
            cleared_negative = metrics['specificity'] * (1.0 - other)
            cleared_positive = miss_rate * other
            log('      prevalence {:>4.0%}  ->  NPV {:.4f}'.format(
                other, cleared_negative / max(cleared_negative + cleared_positive, 1e-9)))

    '''
    Discrimination: threshold-free
    '''
    auc_low, auc_high = bootstrap_interval(
        test_probabilities, test_labels, eval_utils.compute_auc_roc, args.n_bootstrap)

    log('')
    log('=' * 78)
    log('DISCRIMINATION  -  independent of any threshold')
    log('=' * 78)
    log('  Held-out AUC-ROC   {:.5f}   95% CI [{:.4f}, {:.4f}]  ({} resamples)'.format(
        eval_utils.compute_auc_roc(test_probabilities, test_labels),
        auc_low, auc_high, args.n_bootstrap))
    log('  Validation AUC-ROC {:.5f}   (thresholds are calibrated here, never reported from)'.format(
        eval_utils.compute_auc_roc(val_probabilities, val_labels)))

    '''
    Safety: malignancy leakage, and the operating point the budget buys
    '''
    if args.test_original_labels_file is not None:
        test_original = read_column(args.test_original_labels_file, len(test_labels))
        malignant_set = set(args.malignant_labels)
        test_malignant = np.array([label in malignant_set for label in test_original])

        log('')
        log('=' * 78)
        log('SAFETY  -  malignant lesions flagged as eczema')
        log('=' * 78)
        log('  {} malignant lesions in the held-out set'.format(int(test_malignant.sum())))
        log('')
        log('  {:>10s}  {:>11s}  {:>11s}  {:>9s}  {:>16s}'.format(
            'threshold', 'sensitivity', 'specificity', 'PPV', 'malignant flagged'))

        for candidate in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
            metrics = point_metrics(test_probabilities, test_labels, candidate)
            leaked = int((test_probabilities[test_malignant] >= candidate).sum())

            log('  {:>10.2f}  {:>11.3f}  {:>11.3f}  {:>9.3f}  {:>9d} ({:>4.1f}%)'.format(
                candidate, metrics['sensitivity'], metrics['specificity'], metrics['ppv'],
                leaked, 100.0 * leaked / max(test_malignant.sum(), 1)))

        # The operating point implied by the safety budget, calibrated on validation
        if args.val_original_labels_file is not None:
            val_original = read_column(args.val_original_labels_file, len(val_labels))
            val_malignant = np.array([label in malignant_set for label in val_original])

            budget_threshold = leakage_threshold(
                val_probabilities, val_malignant, args.max_malignant_leakage)
            metrics = point_metrics(test_probabilities, test_labels, budget_threshold)
            leaked = int((test_probabilities[test_malignant] >= budget_threshold).sum())

            log('')
            log('  Operating point from the {:.0%} malignancy budget (calibrated on validation):'.format(
                args.max_malignant_leakage))
            log('    Threshold        p >= {:.4g}'.format(budget_threshold))
            log('    Sensitivity      {:.4f}     Specificity  {:.4f}'.format(
                metrics['sensitivity'], metrics['specificity']))
            log('    PPV              {:.4f}     NPV          {:.4f}'.format(
                metrics['ppv'], metrics['npv']))
            log('    Malignant flagged {} of {} ({:.1%}) on held-out'.format(
                leaked, int(test_malignant.sum()),
                leaked / max(test_malignant.sum(), 1)))
            log('')
            log('    Sensitivity here is a consequence of the safety budget, not a target. The')
            log('    model was never trained to detect malignancy, so this is a measured')
            log('    property and not a designed safeguard.')

    '''
    Sensitivity targets: secondary detail
    '''
    log('')
    log('=' * 78)
    log('SENSITIVITY TARGETS  (secondary)')
    log('=' * 78)
    log('  {:>11s}  {:>9s}  {:>11s}  {:>11s}  {:>20s}  {:>7s}  {:>5s} {:>5s} {:>5s} {:>5s}'.format(
        'target sens', 'threshold', 'sensitivity', 'specificity', 'specificity 95% CI',
        'PPV', 'TP', 'FP', 'TN', 'FN'))

    operating_points = []
    thresholds = {}

    for target_sensitivity in args.target_sensitivities:
        _, candidate = eval_utils.specificity_at_sensitivity(
            val_probabilities, val_labels, target_sensitivity)
        thresholds[target_sensitivity] = candidate

        metrics = point_metrics(test_probabilities, test_labels, candidate)
        operating_points.append(
            (target_sensitivity, metrics['sensitivity'], metrics['specificity']))

        # The threshold is held fixed while the cases resample, which is the question being
        # asked: how far would this operating point move on another test set of the same size.
        specificity_low, specificity_high = bootstrap_interval(
            test_probabilities, test_labels,
            lambda p, l, t=candidate: eval_utils.evaluate(
                (p >= t).astype(np.int64), p, l)['specificity'],
            args.n_bootstrap)

        # .4g not .4f: an overconfident model pushes the calibrated threshold down to ~1e-8,
        # which fixed-point formatting renders as a misleading 0.0000.
        log('  {:>11.0%}  {:>9.4g}  {:>11.5f}  {:>11.5f}  {:>9.4f}, {:<9.4f}  {:>7.4f}  {:>5d} {:>5d} {:>5d} {:>5d}'.format(
            target_sensitivity, candidate, metrics['sensitivity'], metrics['specificity'],
            specificity_low, specificity_high, metrics['ppv'],
            metrics['tp'], metrics['fp'], metrics['tn'], metrics['fn']))

    log('')
    log('  Sensitivity is measured on the held-out split, so it lands near but not exactly on')
    log('  the target - the threshold was calibrated on a different, disjoint set of images.')

    headline = min(args.target_sensitivities, key=lambda s: abs(s - 0.90))

    '''
    False positives by true diagnosis
    '''
    if args.test_original_labels_file is not None:
        predictions = (test_probabilities >= thresholds[headline]).astype(np.int64)
        false_positive = (predictions == 1) & (test_labels == 0)

        log('')
        log('=' * 78)
        log('FALSE POSITIVES BY TRUE DIAGNOSIS  at the {:.0%} sensitivity point'.format(headline))
        log('=' * 78)
        log('  {:>6s}  {:>7s}  {:>6s}   {}'.format('FP', 'non-ecz', 'rate', 'true diagnosis'))

        counts = []
        for diagnosis in sorted(set(test_original[test_labels == 0])):
            is_diagnosis = test_original == diagnosis
            n_total = int((is_diagnosis & (test_labels == 0)).sum())
            n_false_positive = int((is_diagnosis & false_positive).sum())

            if n_false_positive > 0:
                counts.append((n_false_positive, n_total, diagnosis))

        # Worst offenders first: this is the table that decides whether a false positive is
        # actually cheap, so the expensive ones should not be buried in an alphabetical list.
        for n_false_positive, n_total, diagnosis in sorted(
                counts, key=lambda row: row[0] / row[1], reverse=True):
            log('  {:>6d}  {:>7d}  {:>5.1f}%   {}{}'.format(
                n_false_positive, n_total, 100.0 * n_false_positive / n_total, diagnosis,
                '   <-- MALIGNANT' if diagnosis in malignant_set else ''))

    '''
    Skin tone
    '''
    if args.test_scales_file is not None:
        scales = read_column(args.test_scales_file, len(test_labels))
        predictions = (test_probabilities >= thresholds[headline]).astype(np.int64)

        log('')
        log('=' * 78)
        log('FITZPATRICK SKIN TYPE  at the {:.0%} sensitivity point'.format(headline))
        log('=' * 78)
        log('  {:>8s}  {:>6s}  {:>7s}  {:>11s}  {:>11s}  {:>8s}  {:>20s}'.format(
            'type', 'n', 'eczema', 'sensitivity', 'specificity', 'AUC-ROC', 'AUC 95% CI'))

        # Types are grouped in pairs: individually, type VI holds too few eczema cases to
        # support any estimate at all.
        for name, members in [('I-II', {'1', '2'}), ('III-IV', {'3', '4'}), ('V-VI', {'5', '6'})]:
            group = np.array([scale in members for scale in scales])

            if group.sum() == 0:
                continue

            group_results = eval_utils.evaluate(
                predictions[group], test_probabilities[group], test_labels[group])

            # The per-group CI is what makes this table readable: the point estimates will differ
            # between groups on sample size alone, and overlapping intervals say so.
            group_low, group_high = bootstrap_interval(
                test_probabilities[group], test_labels[group],
                eval_utils.compute_auc_roc, args.n_bootstrap)

            log('  {:>8s}  {:>6d}  {:>7d}  {:>11.5f}  {:>11.5f}  {:>8.5f}  {:>9.4f}, {:<9.4f}'.format(
                name, int(group.sum()), int(test_labels[group].sum()),
                group_results['recall_per_class'][eval_utils.POSITIVE_CLASS],
                group_results['specificity'], group_results['auc_roc'],
                group_low, group_high))

        n_dark_positive = int(test_labels[np.array([s in {'5', '6'} for s in scales])].sum())
        log('')
        log('  Type V-VI holds {} eczema cases, so its sensitivity moves ~{:.0f} points per case.'.format(
            n_dark_positive, 100.0 / max(n_dark_positive, 1)))
        log('  Read sensitivity as directional only. The AUC column with its interval is the')
        log('  defensible comparison, and overlapping intervals mean no difference was detected -')
        log('  which is not the same as having shown the groups perform equally.')

    if args.output_path is not None:
        os.makedirs(args.output_path, exist_ok=True)

        with open(os.path.join(args.output_path, 'operating_points.txt'), 'w') as f:
            f.write('\n'.join(lines) + '\n')

        fpr, tpr, roc_thresholds = roc_curve(test_labels, test_probabilities)

        # The curve always goes out as CSV so the slide can be built in anything; the PNG is a
        # convenience and matplotlib is not in .venv, so it stays optional.
        np.savetxt(
            os.path.join(args.output_path, 'roc.csv'),
            np.column_stack([fpr, tpr, roc_thresholds]),
            delimiter=',', header='false_positive_rate,sensitivity,threshold', comments='')

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print('\nWrote {0}/operating_points.txt and {0}/roc.csv'.format(args.output_path))
            print('Skipped roc.png: matplotlib is not installed (pip install matplotlib).')
            raise SystemExit(0)

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='#1f77b4', linewidth=2, label='Held-out ROC (AUC {:.3f})'.format(
            eval_utils.compute_auc_roc(test_probabilities, test_labels)))
        plt.plot([0, 1], [0, 1], color='#999999', linestyle='--', linewidth=1, label='Chance')

        for target_sensitivity, sensitivity, specificity in operating_points:
            plt.plot(1.0 - specificity, sensitivity, 'o', color='#d62728', markersize=8)
            plt.annotate(
                '{:.0%} target\nsens {:.3f} / spec {:.3f}'.format(
                    target_sensitivity, sensitivity, specificity),
                xy=(1.0 - specificity, sensitivity),
                xytext=(12, -22), textcoords='offset points', fontsize=8)

        plt.xlabel('False positive rate (1 - specificity)')
        plt.ylabel('Sensitivity (eczema recall)')
        plt.title('Eczema binary classification, held-out set')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_path, 'roc.png'), dpi=150)

        print('\nWrote {0}/operating_points.txt, {0}/roc.csv and {0}/roc.png'.format(
            args.output_path))
