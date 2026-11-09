"""Calibration metrics: ECE, Brier score, reliability binning."""
import numpy as np
from sklearn.metrics import brier_score_loss


def expected_calibration_error(y_true, y_prob, n_bins=15):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Fix: Make the first bin inclusive of 0.0
        if i == 0:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob > lo) & (y_prob <= hi)
            
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def reliability_bins(y_true, y_prob, n_bins=15):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, confs, accs, counts = [], [], [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob > lo) & (y_prob <= hi)
        centers.append((lo + hi) / 2)
        if mask.sum() == 0:
            confs.append(np.nan)
            accs.append(np.nan)
            counts.append(0)
        else:
            confs.append(float(y_prob[mask].mean()))
            accs.append(float(y_true[mask].mean()))
            counts.append(int(mask.sum()))
    return {
        "bin_center": centers,
        "mean_confidence": confs,
        "mean_accuracy": accs,
        "count": counts,
    }


def brier(y_true, y_prob):
    return float(brier_score_loss(y_true, y_prob))