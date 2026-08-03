"""
Synthetic driver-session simulator.

A latent (fatigue, stress) process evolves over each driving session and *drives*
the observable multimodal signals — behavioural (lateral drift, steering
reversals), event history, and physiological (heart rate, HRV, PERCLOS, blink
rate). Fatigue rises with time-on-task, faster at night and on monotonous roads,
and resets after a break; stress spikes around near-misses / hard braking.

Each signal is a noisy function of the latent state, so no single timestep is
reliable — the model must integrate a *window* over time (which is exactly what
the LSTM / Transformer do). Ground-truth fatigue labels let us validate them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Settings, settings as default_settings
from ..signals import FEATURE_COLUMNS

ROAD_TYPES = ("highway", "rural", "urban")
# monotony (fatigue multiplier), base stress, base speed kph
ROAD_SPEC = {
    "highway": (1.35, 0.20, 115),
    "rural":   (1.05, 0.28, 75),
    "urban":   (0.80, 0.42, 45),
}


@dataclass
class Session:
    driver_id: int
    signals: np.ndarray   # [T, N_FEATURES] in FEATURE_COLUMNS order
    fatigue: np.ndarray   # [T] latent, ground truth
    stress: np.ndarray    # [T] latent, ground truth
    labels: np.ndarray    # [T] fatigue class 0/1/2
    road_type: str
    is_night: bool


def _fatigue_class(fatigue, cfg: Settings) -> np.ndarray:
    return (fatigue >= cfg.drowsy_threshold).astype(int) + \
           (fatigue >= cfg.critical_threshold).astype(int)


def _simulate_one(driver_id: int, trait: dict, rng: np.random.Generator,
                  cfg: Settings) -> Session:
    T = int(cfg.session_minutes * 60 * cfg.sample_hz)
    t_min = np.arange(T) / (60.0 * cfg.sample_hz)      # minutes elapsed
    road = rng.choice(ROAD_TYPES, p=[0.4, 0.3, 0.3])
    monotony, base_stress, base_speed = ROAD_SPEC[road]
    is_night = bool(rng.random() < 0.30)

    # --- latent fatigue: drift up with time, faster at night / monotony,
    #     reset by an optional mid-session break, plus a slow random walk ------
    circadian = 1.35 if is_night else 1.0
    rate = 0.0130 * monotony * circadian * trait["fatigue_gain"]
    walk = np.cumsum(rng.normal(0, 0.006, T))
    fatigue = trait["fatigue0"] + rate * t_min + walk
    time_since_break = t_min.copy()
    if rng.random() < 0.4:                              # a break partway through
        b = int(rng.uniform(0.35, 0.7) * T)
        fatigue[b:] -= rng.uniform(0.15, 0.30)
        time_since_break[b:] = t_min[b:] - t_min[b]
    fatigue = np.clip(fatigue, 0.02, 1.0)

    # --- events + stress: baseline + decaying spikes on incidents ------------
    hard_brake = np.zeros(T)
    near_miss = np.zeros(T)
    stress = np.full(T, base_stress * trait["stress_base"])
    for t in range(T):
        p_evt = np.clip(0.004 + 0.02 * fatigue[t] + 0.01 * stress[t], 0, 0.2)
        if rng.random() < p_evt:
            if rng.random() < 0.6:
                hard_brake[t] = 1
                stress[t:t + 30] += 0.25 * np.exp(-np.arange(min(30, T - t)) / 12)
            else:
                near_miss[t] = 1
                stress[t:t + 40] += 0.40 * np.exp(-np.arange(min(40, T - t)) / 15)
    stress = np.clip(stress + rng.normal(0, 0.03, T), 0, 1)

    # rolling 60 s event counts
    k = int(60 * cfg.sample_hz)
    kernel = np.ones(k)
    recent_hb = np.convolve(hard_brake, kernel, "full")[:T]
    recent_nm = np.convolve(near_miss, kernel, "full")[:T]

    f, s = fatigue, stress
    n = lambda sd: rng.normal(0, sd, T)                # noise helper
    ch = {
        # behavioural
        "lateral_deviation_m": np.clip(0.12 + 0.55 * f + 0.10 * s + n(0.06), 0, None),
        "steering_reversal_rate": np.clip(7 + 22 * f + 9 * s + n(2.5), 0, None),
        "lane_offset_std": np.clip(0.08 + 0.42 * f + n(0.05), 0, None),
        "time_headway_s": np.clip(2.1 - 0.9 * s + 0.4 * (f - 0.5) + n(0.25), 0.4, 6),
        "speed_kph": np.clip(base_speed * (1 - 0.08 * f) + n(4.0), 0, None),
        "throttle_brake_jerk": np.clip(0.18 + 0.5 * s + 0.3 * f + n(0.06), 0, None),
        # event history
        "recent_hard_brakes": recent_hb,
        "recent_near_misses": recent_nm,
        # context
        "time_on_task_min": t_min,
        "time_since_break_min": time_since_break,
        "is_night": np.full(T, float(is_night)),
        # physiological
        "heart_rate_bpm": np.clip(trait["hr0"] + 30 * s - 6 * f + n(2.5), 40, 180),
        "hrv_sdnn_ms": np.clip(58 - 32 * s + 6 * (1 - f) + n(4.0), 8, 120),
        "blink_rate_per_min": np.clip(9 + 20 * f + n(2.0), 0, None),
        "perclos": np.clip(0.02 + 0.40 * f ** 1.5 + n(0.03), 0, 1),
        "gaze_off_road_ratio": np.clip(0.04 + 0.30 * f + 0.10 * s + n(0.04), 0, 1),
    }

    signals = np.stack([ch[c] for c in FEATURE_COLUMNS], axis=1).astype(np.float32)
    labels = _fatigue_class(fatigue, cfg)
    return Session(driver_id, signals, fatigue, stress, labels, road, is_night)


def generate_sessions(cfg: Settings | None = None) -> list[Session]:
    cfg = cfg or default_settings
    rng = np.random.default_rng(cfg.seed)
    sessions: list[Session] = []
    for d in range(cfg.n_drivers):
        trait = {
            "fatigue0": rng.uniform(0.02, 0.15),
            "fatigue_gain": rng.uniform(0.75, 1.35),   # some drivers tire faster
            "stress_base": rng.uniform(0.7, 1.3),
            "hr0": rng.uniform(60, 78),
        }
        for _ in range(cfg.sessions_per_driver):
            sessions.append(_simulate_one(d, trait, rng, cfg))
    return sessions


def save_demo_sessions(cfg: Settings | None = None, n: int = 6) -> None:
    """Save a handful of full sessions for the dashboard to replay."""
    cfg = cfg or default_settings
    cfg.ensure_dirs()
    sessions = generate_sessions(cfg)[:n]
    np.savez_compressed(
        cfg.data_root / "demo_sessions.npz",
        signals=np.stack([s.signals for s in sessions]),
        fatigue=np.stack([s.fatigue for s in sessions]),
        stress=np.stack([s.stress for s in sessions]),
        labels=np.stack([s.labels for s in sessions]),
        road_type=np.array([s.road_type for s in sessions]),
        is_night=np.array([s.is_night for s in sessions]),
    )
