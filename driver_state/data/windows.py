"""
Windowing + dataset assembly.

Turns raw sessions into fixed-length sliding windows. Each window is labelled
with the fatigue class at its *last* timestep — i.e. "given the last 60 s of
signals, what is the driver's state now?". Splits are made by *driver* so no
driver appears in both train and test (prevents optimistic leakage). Features are
standardised using train-set statistics only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from ..config import Settings, settings as default_settings
from ..signals import FEATURE_COLUMNS


@dataclass
class StandardScaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["mean"], d["std"])


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler


def _windows_from_session(signals, labels, win, stride):
    xs, ys = [], []
    for start in range(0, len(signals) - win + 1, stride):
        xs.append(signals[start:start + win])
        ys.append(labels[start + win - 1])          # state at window end
    return xs, ys


def build_dataset(sessions, cfg: Settings | None = None) -> Dataset:
    cfg = cfg or default_settings
    win, stride = cfg.window_len, cfg.window_stride

    rng = np.random.default_rng(cfg.seed)
    drivers = sorted({s.driver_id for s in sessions})
    rng.shuffle(drivers)
    n = len(drivers)
    train_d = set(drivers[: int(0.70 * n)])
    val_d = set(drivers[int(0.70 * n): int(0.85 * n)])

    buckets = {"train": ([], []), "val": ([], []), "test": ([], [])}
    for s in sessions:
        split = "train" if s.driver_id in train_d else \
                "val" if s.driver_id in val_d else "test"
        xs, ys = _windows_from_session(s.signals, s.labels, win, stride)
        buckets[split][0].extend(xs)
        buckets[split][1].extend(ys)

    def arr(split):
        X = np.asarray(buckets[split][0], dtype=np.float32)
        y = np.asarray(buckets[split][1], dtype=np.int64)
        return X, y

    X_train, y_train = arr("train")
    X_val, y_val = arr("val")
    X_test, y_test = arr("test")

    # standardise on train statistics (per feature, over all timesteps)
    flat = X_train.reshape(-1, X_train.shape[-1])
    scaler = StandardScaler(flat.mean(0), flat.std(0) + 1e-6)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return Dataset(X_train, y_train, X_val, y_val, X_test, y_test, scaler)


def save_feature_meta(cfg: Settings | None = None) -> None:
    cfg = cfg or default_settings
    cfg.ensure_dirs()
    (cfg.model_root / "features.json").write_text(json.dumps({
        "feature_columns": FEATURE_COLUMNS,
        "window_len": cfg.window_len,
    }, indent=2))
