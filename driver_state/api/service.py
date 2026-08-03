"""
Serving layer: loads the model + demo sessions, answers state/policy queries,
replays sessions for the dashboard, and reports calibration + drift.
"""
from __future__ import annotations

import json

import numpy as np

from ..config import Settings, settings as default_settings
from ..models.infer import DriverStateModel
from ..monitoring.metrics import ks_statistic, psi
from ..policy import decide_policy
from ..signals import CLASS_NAMES, FEATURE_COLUMNS, N_FEATURES

DRIFT_FEATURES = ["perclos", "lateral_deviation_m", "heart_rate_bpm",
                  "steering_reversal_rate"]


class DriverStateService:
    def __init__(self, cfg: Settings | None = None):
        self.cfg = cfg or default_settings
        self.model = DriverStateModel.load(self.cfg)
        self.metrics = json.loads((self.cfg.model_root / "metrics.json").read_text())
        d = np.load(self.cfg.data_root / "demo_sessions.npz", allow_pickle=True)
        self.demo = {k: d[k] for k in d.files}
        self._fi = {c: i for i, c in enumerate(FEATURE_COLUMNS)}

    # ---------------------------------------------------------------- predict
    def _check(self, window):
        w = np.asarray(window, dtype=np.float32)
        if w.ndim != 2 or w.shape[1] != N_FEATURES:
            raise ValueError(f"window must be [T, {N_FEATURES}], got {w.shape}")
        if w.shape[0] != self.cfg.window_len:
            raise ValueError(f"window length must be {self.cfg.window_len}, "
                             f"got {w.shape[0]}")
        return w

    def predict(self, window) -> dict:
        return self.model.predict_window(self._check(window))

    def policy(self, window, is_night=False, time_on_task_min=None,
               stress=None) -> dict:
        state = self.predict(window)
        pol = decide_policy(state["fatigue_state"], state["fatigue_score"],
                            stress=stress, time_on_task_min=time_on_task_min,
                            is_night=is_night)
        return {"driver_state": state, "policy": pol.as_dict()}

    # ------------------------------------------------------------- dashboard
    def n_demo(self) -> int:
        return int(self.demo["signals"].shape[0])

    def simulate_session(self, index: int = 0) -> dict:
        n = self.n_demo()
        index = int(index) % n
        signals = self.demo["signals"][index]           # [T, F]
        fatigue = self.demo["fatigue"][index]            # [T]
        labels = self.demo["labels"][index]              # [T]
        is_night = bool(self.demo["is_night"][index])
        road = str(self.demo["road_type"][index])
        win, stride = self.cfg.window_len, self.cfg.window_stride

        times, pred_score, pred_state, gt_fatigue, gt_state = [], [], [], [], []
        sig_out = {f: [] for f in ("perclos", "lateral_deviation_m", "heart_rate_bpm")}
        correct = 0
        ends = list(range(win - 1, len(signals), stride))
        for e in ends:
            out = self.model.predict_window(signals[e - win + 1: e + 1])
            times.append(round(e / 60.0, 2))
            pred_score.append(round(out["fatigue_score"], 3))
            pred_state.append(out["fatigue_state"])
            gt_fatigue.append(round(float(fatigue[e]), 3))
            gt_state.append(CLASS_NAMES[int(labels[e])])
            correct += int(out["fatigue_state"] == CLASS_NAMES[int(labels[e])])
            for f in sig_out:
                sig_out[f].append(round(float(signals[e, self._fi[f]]), 3))

        last = self.model.predict_window(signals[ends[-1] - win + 1: ends[-1] + 1])
        pol = decide_policy(last["fatigue_state"], last["fatigue_score"],
                            time_on_task_min=times[-1], is_night=is_night)
        return {
            "index": index, "road_type": road, "is_night": is_night,
            "n_windows": len(ends),
            "session_accuracy": round(correct / len(ends), 3),
            "times_min": times, "pred_score": pred_score, "pred_state": pred_state,
            "gt_fatigue": gt_fatigue, "gt_state": gt_state, "signals": sig_out,
            "current_state": last, "current_policy": pol.as_dict(),
        }

    # --------------------------------------------------------------- metrics
    def metrics_report(self) -> dict:
        sig = self.demo["signals"].reshape(-1, N_FEATURES)
        fat = self.demo["fatigue"].reshape(-1)
        ref = sig[fat < 0.33]                              # alert cohort
        cur = sig[fat > 0.55]                              # fatigued cohort
        drift = {}
        for f in DRIFT_FEATURES:
            i = self._fi[f]
            p = psi(ref[:, i], cur[:, i])
            drift[f] = {
                "psi": round(p, 3),
                "ks": round(ks_statistic(ref[:, i], cur[:, i]), 3),
                "verdict": ("significant" if p > 0.25 else
                            "moderate" if p > 0.1 else "stable"),
            }
        return {
            "models": self.metrics["models"],
            "serving_model": self.metrics["serving_model"],
            "reliability": self.metrics["reliability"],
            "drift_alert_vs_fatigued": drift,
            "drift_note": "Population-stability (PSI) & KS between alert and "
                          "fatigued cohorts — large values confirm these signals "
                          "genuinely separate the states the model keys on.",
        }


_service: DriverStateService | None = None


def get_service(cfg: Settings | None = None) -> DriverStateService:
    global _service
    if _service is None:
        _service = DriverStateService(cfg)
    return _service
