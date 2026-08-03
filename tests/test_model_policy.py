"""Model inference + policy engine tests."""
from __future__ import annotations

import json

import numpy as np

from driver_state.models.infer import DriverStateModel
from driver_state.policy import decide_policy


def test_training_beats_chance(cfg):
    metrics = json.loads((cfg.model_root / "metrics.json").read_text())
    for m in metrics["models"].values():
        assert m["accuracy"] > 0.45          # >> 1/3 chance for 3 classes
        assert 0.0 <= m["ece"] <= 1.0
    assert metrics["serving_model"] in ("lstm", "transformer")


def test_predict_window(cfg):
    model = DriverStateModel.load(cfg)
    window = np.random.default_rng(0).normal(size=(cfg.window_len,
                                                    model.scaler.mean.shape[0]))
    out = model.predict_window(window)
    assert out["fatigue_state"] in ("alert", "drowsy", "critical")
    assert 0.0 <= out["fatigue_score"] <= 1.0
    assert abs(sum(out["probabilities"].values()) - 1.0) < 1e-4


def test_policy_monotonic():
    low = decide_policy("alert", 0.1)
    high = decide_policy("critical", 0.9)
    assert high.fcw_lead_time_s > low.fcw_lead_time_s
    assert high.following_distance_s > low.following_distance_s
    assert high.alert_volume > low.alert_volume
    order = {"standard": 0, "firm": 1, "max": 2}
    assert order[high.lane_keep_strictness] > order[low.lane_keep_strictness]
    assert high.break_recommendation and not low.break_recommendation


def test_policy_night_caps_brightness():
    day = decide_policy("critical", 0.9, is_night=False)
    night = decide_policy("critical", 0.9, is_night=True)
    assert night.display_brightness <= day.display_brightness
