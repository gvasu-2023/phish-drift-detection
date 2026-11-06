import numpy as np
from phishing_detector.calibration.metrics import (
    expected_calibration_error, reliability_bins, brier,
)


def test_ece_perfectly_calibrated_is_zero():
    y_true = np.array([0, 1] * 500)
    y_prob = np.full(1000, 0.5)
    assert expected_calibration_error(y_true, y_prob) < 0.02


def test_ece_worst_is_high():
    y_true = np.zeros(1000, dtype=int)
    y_prob = np.ones(1000)
    assert expected_calibration_error(y_true, y_prob) > 0.95


def test_brier_range():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.4, 0.6])
    assert 0.0 <= brier(y_true, y_prob) <= 1.0


def test_reliability_bins_sum_to_n():
    y_true = np.array([0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.6, 0.3, 0.7])
    out = reliability_bins(y_true, y_prob, n_bins=5)
    assert sum(out["count"]) == len(y_true)