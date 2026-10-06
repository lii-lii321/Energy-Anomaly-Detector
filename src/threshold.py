import numpy as np


def compute_prf(y_true, y_pred):
    """计算 precision/recall/F1；对应分母为零时该指标记 0。"""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.size == 0 or y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must be non-empty arrays of equal length")
    if not np.all(np.isin(y_true, (0, 1))) or not np.all(np.isin(y_pred, (0, 1))):
        raise ValueError("y_true and y_pred must contain only 0 and 1")

    true_positive = int(np.sum((y_true == 1) & (y_pred == 1)))
    false_positive = int(np.sum((y_true == 0) & (y_pred == 1)))
    false_negative = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_f1(y_true, y_pred):
    return compute_prf(y_true, y_pred)["f1"]


def find_best_threshold(y_true, probs):
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    if y_true.size == 0 or y_true.shape != probs.shape:
        raise ValueError("y_true and probs must be non-empty arrays of equal length")
    if not np.all(np.isin(y_true, (0, 1))):
        raise ValueError("y_true must contain only 0 and 1")

    best_threshold, best_f1 = 0.0, -1.0
    for threshold in np.unique(probs):
        f1 = compute_f1(y_true, probs < threshold)
        if f1 > best_f1:
            best_threshold, best_f1 = float(threshold), f1
    return best_threshold, best_f1
