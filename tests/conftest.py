"""Shared fixture: train a tiny model in a temp dir once per test session."""
from __future__ import annotations

import pytest

from driver_state.config import Settings
from driver_state.generation import save_demo_sessions
from driver_state.models.train import train_all


@pytest.fixture(scope="session")
def cfg(tmp_path_factory) -> Settings:
    d = tmp_path_factory.mktemp("run")
    cfg = Settings(
        data_root=d / "data", model_root=d / "models",
        n_drivers=10, sessions_per_driver=2, epochs=1, seed=1,
    )
    cfg.ensure_dirs()
    train_all(cfg, verbose=False)
    save_demo_sessions(cfg, n=4)
    return cfg
