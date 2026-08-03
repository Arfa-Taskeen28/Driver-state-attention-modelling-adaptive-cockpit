"""API tests — drive every endpoint through the ASGI app with test artifacts."""
from __future__ import annotations

import numpy as np
import pytest
from starlette.testclient import TestClient

from driver_state.api import service as service_mod
from driver_state.api.service import DriverStateService


@pytest.fixture(scope="module")
def client(cfg):
    service_mod._service = DriverStateService(cfg)
    from driver_state.api.main import app
    return TestClient(app)


@pytest.fixture(scope="module")
def window(cfg):
    d = np.load(cfg.data_root / "demo_sessions.npz", allow_pickle=True)
    return d["signals"][0][: cfg.window_len].tolist()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["serving_model"] in ("lstm", "transformer")


def test_schema(client):
    r = client.get("/schema").json()
    assert len(r["feature_columns"]) == r["n_features"]


def test_driver_state(client, window):
    r = client.post("/driver_state", json={"window": window})
    assert r.status_code == 200
    body = r.json()
    assert body["fatigue_state"] in ("alert", "drowsy", "critical")
    assert 0.0 <= body["fatigue_score"] <= 1.0


def test_driver_state_bad_window(client):
    r = client.post("/driver_state", json={"window": [[0.0, 1.0]]})
    assert r.status_code == 422


def test_policy(client, window):
    r = client.post("/policy", json={"window": window, "is_night": True,
                                     "time_on_task_min": 130})
    assert r.status_code == 200
    body = r.json()
    assert "policy" in body and "driver_state" in body
    assert body["policy"]["lane_keep_strictness"] in ("standard", "firm", "max")


def test_session_simulate(client):
    r = client.get("/session/simulate", params={"index": 0}).json()
    assert r["n_windows"] > 0
    assert len(r["times_min"]) == len(r["pred_score"])
    assert "current_policy" in r


def test_metrics(client):
    r = client.get("/metrics").json()
    assert "models" in r and "drift_alert_vs_fatigued" in r
