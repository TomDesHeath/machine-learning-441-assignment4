"""
Pure Python & NumPy Performance Metrics for Multi-Class Evaluation.
"""

import numpy as np


def confusion_matrix(y_true, y_pred, num_classes=10):
    """
    Compute confusion matrix where rows represent true classes and columns represent predicted classes.

    Parameters:
    -----------
    y_true : np.ndarray (N,)
        True class labels (0 to num_classes-1).
    y_pred : np.ndarray (N,)
        Predicted class labels.
    num_classes : int
        Total number of target classes.

    Returns:
    --------
    np.ndarray (num_classes, num_classes)
        Confusion matrix C where C[i, j] is number of samples with true class i predicted as j.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[int(t), int(p)] += 1
    return cm


def accuracy_score(y_true, y_pred):
    """
    Compute overall classification accuracy.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)


def balanced_accuracy_score(y_true, y_pred, num_classes=10):
    """
    Compute balanced accuracy score (unweighted mean of recall per class).
    """
    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    recalls = []
    for c in range(num_classes):
        total_true = np.sum(cm[c, :])
        if total_true > 0:
            recalls.append(cm[c, c] / total_true)
        else:
            recalls.append(0.0)
    return float(np.mean(recalls))


def precision_recall_f1_score(y_true, y_pred, average='macro', num_classes=10):
    """
    Compute Precision, Recall, and F1-score across classes.

    Parameters:
    -----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    average : str ('macro', 'weighted', or None)
        Averaging method.
    num_classes : int
        Number of target classes.

    Returns:
    --------
    tuple (precision, recall, f1) or dict if average is None
    """
    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    precisions = np.zeros(num_classes, dtype=np.float64)
    recalls = np.zeros(num_classes, dtype=np.float64)
    f1s = np.zeros(num_classes, dtype=np.float64)
    support = np.zeros(num_classes, dtype=np.float64)

    for c in range(num_classes):
        tp = cm[c, c]
        fn = np.sum(cm[c, :]) - tp
        fp = np.sum(cm[:, c]) - tp

        support[c] = tp + fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions[c] = prec
        recalls[c] = rec
        f1s[c] = f1

    if average is None:
        return precisions, recalls, f1s, support

    total_samples = np.sum(support)
    if average == 'macro':
        p_avg = float(np.mean(precisions))
        r_avg = float(np.mean(recalls))
        f1_avg = float(np.mean(f1s))
    elif average == 'weighted':
        if total_samples > 0:
            p_avg = float(np.sum(precisions * support) / total_samples)
            r_avg = float(np.sum(recalls * support) / total_samples)
            f1_avg = float(np.sum(f1s * support) / total_samples)
        else:
            p_avg, r_avg, f1_avg = 0.0, 0.0, 0.0
    else:
        raise ValueError(f"Unknown average mode: {average}")

    return p_avg, r_avg, f1_avg


def cohens_kappa_score(y_true, y_pred, num_classes=10):
    """
    Compute Cohen's Kappa score for multi-class agreement.
    """
    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    total = np.sum(cm)
    if total == 0:
        return 0.0

    p_o = np.trace(cm) / total

    sum_row = np.sum(cm, axis=1)
    sum_col = np.sum(cm, axis=0)

    p_e = np.sum(sum_row * sum_col) / (total ** 2)

    if p_e == 1.0:
        return 1.0

    kappa = (p_o - p_e) / (1.0 - p_e)
    return float(kappa)


def get_all_metrics(y_true, y_pred, num_classes=10):
    """
    Compute a full dictionary of multi-class metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred, num_classes=num_classes)
    p_mac, r_mac, f1_mac = precision_recall_f1_score(y_true, y_pred, average='macro', num_classes=num_classes)
    p_w, r_w, f1_w = precision_recall_f1_score(y_true, y_pred, average='weighted', num_classes=num_classes)
    kappa = cohens_kappa_score(y_true, y_pred, num_classes=num_classes)

    return {
        'accuracy': float(acc),
        'balanced_accuracy': float(bal_acc),
        'macro_precision': float(p_mac),
        'macro_recall': float(r_mac),
        'macro_f1': float(f1_mac),
        'weighted_f1': float(f1_w),
        'cohens_kappa': float(kappa)
    }
