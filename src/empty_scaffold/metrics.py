from __future__ import annotations

import numpy as np


def _finite_pair_arrays(y_true, y_score):
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    return y_true[mask], y_score[mask]


def _average_precision_grouped(y_true, y_score) -> float:
    """Average precision with score ties handled as threshold groups."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-y_score, kind="mergesort")
    y = y_true[order]
    scores = y_score[order]
    group_end = np.r_[np.where(np.diff(scores) != 0)[0], len(scores) - 1]
    tp_cum = np.cumsum(y == 1)[group_end]
    fp_cum = (group_end + 1) - tp_cum
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    recall = tp_cum / n_pos
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _roc_auc_rank(y_true, y_score) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(y_score)
    sum_pos = float(np.sum(ranks[y_true == 1]))
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _binary_threshold_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if len(y_true) == 0:
        return {"accuracy": float("nan"), "balanced_accuracy": float("nan"), "f1": float("nan"), "mcc": float("nan")}

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    acc = (tp + tn) / len(y_true)
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")
    bal = np.nanmean([tpr, tnr])
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / np.sqrt(denom)) if denom > 0 else float("nan")
    return {"accuracy": float(acc), "balanced_accuracy": float(bal), "f1": float(f1), "mcc": float(mcc)}


def norm_pr(y_true, y_score) -> float:
    """Average precision normalized against the positive-rate baseline."""
    y_true, y_score = _finite_pair_arrays(y_true, y_score)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    y_true = y_true.astype(int)
    pos_rate = float(np.mean(y_true == 1))
    if pos_rate >= 1.0:
        return float("nan")
    ap = _average_precision_grouped(y_true, y_score)
    return float((ap - pos_rate) / (1.0 - pos_rate))


def classification_summary(y_true, y_score, y_pred=None) -> dict:
    y_true, y_score = _finite_pair_arrays(y_true, y_score)
    if y_pred is None:
        y_pred = (y_score >= 0.5).astype(int) if len(y_score) else np.asarray([])
    else:
        _, y_pred = _finite_pair_arrays(y_true, y_pred)
    out = {
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true == 1)) if len(y_true) else float("nan"),
        "average_precision": float("nan"),
        "roc_auc": float("nan"),
        "normPR": float("nan"),
        "accuracy": float("nan"),
        "balanced_accuracy": float("nan"),
        "f1": float("nan"),
        "mcc": float("nan"),
    }
    if len(y_true):
        y_true_i = y_true.astype(int)
        out.update(_binary_threshold_metrics(y_true_i, np.asarray(y_pred).astype(int)))
        if len(np.unique(y_true_i)) >= 2:
            out["average_precision"] = _average_precision_grouped(y_true_i, y_score)
            out["roc_auc"] = _roc_auc_rank(y_true_i, y_score)
            out["normPR"] = norm_pr(y_true_i, y_score)
    return out


def rel_mae(y_true, y_pred) -> float:
    y_true, y_pred = _finite_pair_arrays(y_true, y_pred)
    if len(y_true) == 0:
        return float("nan")
    scale = float(np.mean(np.abs(y_true)))
    if scale == 0.0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def _pearson(x, y) -> float:
    x, y = _finite_pair_arrays(x, y)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x, y) -> float:
    x, y = _finite_pair_arrays(x, y)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return _pearson(_average_ranks(x), _average_ranks(y))


def regression_summary(y_true, y_pred) -> dict:
    y_true, y_pred = _finite_pair_arrays(y_true, y_pred)
    out = {
        "n": int(len(y_true)),
        "MAE": float("nan"),
        "RMSE": float("nan"),
        "R2": float("nan"),
        "Pearson_r": float("nan"),
        "Spearman_rho": float("nan"),
        "target_scale_mean_abs": float("nan"),
        "relMAE": float("nan"),
    }
    if len(y_true):
        residual = y_true - y_pred
        out["MAE"] = float(np.mean(np.abs(residual)))
        out["RMSE"] = float(np.sqrt(np.mean(residual ** 2)))
        denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
        out["R2"] = float(1.0 - np.sum(residual ** 2) / denom) if denom > 0 and len(y_true) >= 2 else float("nan")
        out["Pearson_r"] = _pearson(y_true, y_pred)
        out["Spearman_rho"] = _spearman(y_true, y_pred)
        out["target_scale_mean_abs"] = float(np.mean(np.abs(y_true)))
        out["relMAE"] = rel_mae(y_true, y_pred)
    return out
