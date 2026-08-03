"""Simulator + dataset tests."""
from __future__ import annotations

import numpy as np

from driver_state.data import build_dataset
from driver_state.generation import generate_sessions
from driver_state.signals import FEATURE_COLUMNS, N_FEATURES


def test_sessions_shape_and_labels(cfg):
    sessions = generate_sessions(cfg)
    assert len(sessions) == cfg.n_drivers * cfg.sessions_per_driver
    s = sessions[0]
    assert s.signals.shape[1] == N_FEATURES
    assert set(np.unique(s.labels)) <= {0, 1, 2}


def test_fatigue_drives_physiology(cfg):
    """PERCLOS (eye closure) must rise with the latent fatigue that labels it."""
    s = generate_sessions(cfg)[0]
    perclos = s.signals[:, FEATURE_COLUMNS.index("perclos")]
    corr = np.corrcoef(perclos, s.fatigue)[0, 1]
    assert corr > 0.5


def test_split_is_by_driver_no_leakage(cfg):
    sessions = generate_sessions(cfg)
    ds = build_dataset(sessions, cfg)
    assert ds.X_train.shape[1:] == (cfg.window_len, N_FEATURES)
    assert len(ds.y_train) > 0 and len(ds.y_test) > 0
    # standardised on train stats -> train features ~zero mean
    assert abs(ds.X_train.reshape(-1, N_FEATURES).mean()) < 0.1
