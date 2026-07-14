import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix as sk_confusion_matrix
)


def compute_accuracy(predictions, labels):
    '''
    Computes overall classification accuracy

    Args:
        predictions : numpy[int]
            predicted class labels (0 or 1)
        labels : numpy[int]
            ground truth class labels (0 or 1)
    Returns:
        float : accuracy in [0, 1]
    '''

    return accuracy_score(labels, predictions)


def compute_precision(predictions, labels):
    '''
    Computes per-class and macro precision

    Args:
        predictions : numpy[int]
            predicted class labels (0 or 1)
        labels : numpy[int]
            ground truth class labels (0 or 1)
    Returns:
        dict : {
            'precision_per_class': numpy array of per-class precision,
            'precision_macro': float macro-averaged precision
        }
    '''

    per_class = precision_score(labels, predictions, average=None, zero_division=0)
    macro = precision_score(labels, predictions, average='macro', zero_division=0)

    return {
        'precision_per_class': per_class,
        'precision_macro': macro
    }


def compute_recall(predictions, labels):
    '''
    Computes per-class and macro recall (sensitivity)

    Args:
        predictions : numpy[int]
            predicted class labels (0 or 1)
        labels : numpy[int]
            ground truth class labels (0 or 1)
    Returns:
        dict : {
            'recall_per_class': numpy array of per-class recall,
            'recall_macro': float macro-averaged recall
        }
    '''

    per_class = recall_score(labels, predictions, average=None, zero_division=0)
    macro = recall_score(labels, predictions, average='macro', zero_division=0)

    return {
        'recall_per_class': per_class,
        'recall_macro': macro
    }


def compute_f1(predictions, labels):
    '''
    Computes per-class and macro F1 score

    Args:
        predictions : numpy[int]
            predicted class labels (0 or 1)
        labels : numpy[int]
            ground truth class labels (0 or 1)
    Returns:
        dict : {
            'f1_per_class': numpy array of per-class F1,
            'f1_macro': float macro-averaged F1
        }
    '''

    per_class = f1_score(labels, predictions, average=None, zero_division=0)
    macro = f1_score(labels, predictions, average='macro', zero_division=0)

    return {
        'f1_per_class': per_class,
        'f1_macro': macro
    }


def compute_specificity(predictions, labels):
    '''
    Computes specificity (true negative rate) for the positive class

    Args:
        predictions : numpy[int]
            predicted class labels (0 or 1)
        labels : numpy[int]
            ground truth class labels (0 or 1)
    Returns:
        float : specificity (TNR) for the positive class
    '''

    cm = sk_confusion_matrix(labels, predictions, labels=[0, 1])

    # For binary: TN = cm[0,0], FP = cm[0,1]
    tn = cm[0, 0]
    fp = cm[0, 1]

    if (tn + fp) == 0:
        return 0.0

    return float(tn) / float(tn + fp)


def compute_auc_roc(probabilities, labels):
    '''
    Computes area under the ROC curve

    Args:
        probabilities : numpy[float]
            predicted probabilities for the positive class
        labels : numpy[int]
            ground truth class labels (0 or 1)
    Returns:
        float : AUC-ROC score, or -1.0 if only one class is present
    '''

    # AUC is undefined if only one class is present
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return -1.0

    return roc_auc_score(labels, probabilities)


def compute_confusion_matrix(predictions, labels):
    '''
    Computes confusion matrix counts

    Args:
        predictions : numpy[int]
            predicted class labels (0 or 1)
        labels : numpy[int]
            ground truth class labels (0 or 1)
    Returns:
        dict : {
            'tp': int, 'fp': int, 'tn': int, 'fn': int,
            'confusion_matrix': numpy 2x2 array
        }
    '''

    cm = sk_confusion_matrix(labels, predictions, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    return {
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'confusion_matrix': cm
    }


def evaluate(predictions, probabilities, labels):
    '''
    Convenience function that computes all classification metrics at once

    Args:
        predictions : numpy[int]
            predicted class labels (0 or 1)
        probabilities : numpy[float]
            predicted probabilities for the positive class
        labels : numpy[int]
            ground truth class labels (0 or 1)
    Returns:
        dict : dictionary containing all evaluation metrics
    '''

    results = {}

    # Accuracy
    results['accuracy'] = compute_accuracy(predictions, labels)

    # Precision
    precision_results = compute_precision(predictions, labels)
    results['precision_macro'] = precision_results['precision_macro']
    results['precision_per_class'] = precision_results['precision_per_class']

    # Recall
    recall_results = compute_recall(predictions, labels)
    results['recall_macro'] = recall_results['recall_macro']
    results['recall_per_class'] = recall_results['recall_per_class']

    # F1
    f1_results = compute_f1(predictions, labels)
    results['f1_macro'] = f1_results['f1_macro']
    results['f1_per_class'] = f1_results['f1_per_class']

    # Specificity
    results['specificity'] = compute_specificity(predictions, labels)

    # AUC-ROC
    results['auc_roc'] = compute_auc_roc(probabilities, labels)

    # Confusion matrix
    cm_results = compute_confusion_matrix(predictions, labels)
    results['tp'] = cm_results['tp']
    results['fp'] = cm_results['fp']
    results['tn'] = cm_results['tn']
    results['fn'] = cm_results['fn']
    results['confusion_matrix'] = cm_results['confusion_matrix']

    return results
