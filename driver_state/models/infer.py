"""Serving-side model wrapper: raw signal window -> driver-state estimate."""
from __future__ import annotations

import json

import numpy as np
import torch

from ..config import Settings, settings as default_settings
from ..data.windows import StandardScaler
from ..signals import CLASS_NAMES
from . import MODELS

# expected-severity weights: alert=0, drowsy=0.5, critical=1.0
SEVERITY = np.array([0.0, 0.5, 1.0])


class DriverStateModel:
    def __init__(self, model, scaler: StandardScaler, name: str):
        self.model = model.eval()
        self.scaler = scaler
        self.name = name

    @classmethod
    def load(cls, cfg: Settings | None = None) -> "DriverStateModel":
        cfg = cfg or default_settings
        name = json.loads((cfg.model_root / "serving_model.json").read_text())["name"]
        model = MODELS[name]()
        model.load_state_dict(torch.load(cfg.model_root / f"{name}.pt",
                                         map_location="cpu"))
        scaler = StandardScaler.load(cfg.model_root / "scaler.npz")
        return cls(model, scaler, name)

    @torch.no_grad()
    def predict_window(self, window: np.ndarray) -> dict:
        """window: [T, N_FEATURES] raw signals -> driver-state estimate."""
        x = self.scaler.transform(np.asarray(window, dtype=np.float32))[None]
        probs = torch.softmax(self.model(torch.from_numpy(x)), dim=1)[0].numpy()
        cls = int(probs.argmax())
        return {
            "fatigue_class": cls,
            "fatigue_state": CLASS_NAMES[cls],
            "fatigue_score": float(probs @ SEVERITY),        # 0..1 severity
            "confidence": float(probs.max()),
            "probabilities": {c: float(p) for c, p in zip(CLASS_NAMES, probs)},
            "model": self.name,
        }
