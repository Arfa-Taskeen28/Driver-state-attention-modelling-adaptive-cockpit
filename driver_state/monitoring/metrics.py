"""
Model-monitoring metrics — calibration and drift.

A driver-monitoring model that will trigger warnings must be *calibrated* (its
"70% drowsy" should be right ~70% of the time) and watched for *drift* (does the
live signal distribution still match training?). These are the metrics a
production ADAS team tracks; here they're computed in-process and exposed via the
API (an EvidentlyAI dashboard would be the heavier managed alternative).
"""
from __future__ import annotations

import numpy as np


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                               n_bins: int = 10) -> float:
    """ECE over the predicted top-class confidence."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10):
    """Per-bin (confidence, accuracy, count) for a reliability diagram."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        out.append({
            "bin": round((lo + hi) / 2, 3),
            "confidence": float(conf[m].mean()) if m.any() else None,
            "accuracy": float(correct[m].mean()) if m.any() else None,
            "count": int(m.sum()),
        })
    return out


def psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index between two 1-D distributions.
    <0.1 no drift, 0.1-0.25 moderate, >0.25 significant."""
    edges = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_p = np.histogram(reference, edges)[0] / len(reference) + 1e-6
    cur_p = np.histogram(current, edges)[0] / len(current) + 1e-6
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic (max CDF gap) between two samples."""
    allv = np.sort(np.concatenate([reference, current]))
    cdf_r = np.searchsorted(np.sort(reference), allv, "right") / len(reference)
    cdf_c = np.searchsorted(np.sort(current), allv, "right") / len(current)
    return float(np.max(np.abs(cdf_r - cdf_c)))
