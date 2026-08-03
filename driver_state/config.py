"""Central configuration (env-overridable via FLEET-style ``DS_`` variables)."""
from __future__ import annotations

import pathlib

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DS_", env_file=".env",
                                      extra="ignore")

    data_root: pathlib.Path = REPO_ROOT / "data"
    model_root: pathlib.Path = REPO_ROOT / "models"

    # --- Synthetic fleet of drivers -----------------------------------------
    n_drivers: int = 80
    sessions_per_driver: int = 3
    session_minutes: int = 45
    sample_hz: float = 1.0          # signal sampling rate
    seed: int = 7

    # --- Windowing -----------------------------------------------------------
    window_seconds: int = 60        # model input length
    window_stride_seconds: int = 15 # hop between windows

    # --- Fatigue class thresholds (on latent fatigue in [0,1]) --------------
    drowsy_threshold: float = 0.33
    critical_threshold: float = 0.66

    # --- Training ------------------------------------------------------------
    batch_size: int = 128
    epochs: int = 8
    lr: float = 1e-3
    device: str = "cpu"

    @property
    def window_len(self) -> int:
        return int(self.window_seconds * self.sample_hz)

    @property
    def window_stride(self) -> int:
        return int(self.window_stride_seconds * self.sample_hz)

    def ensure_dirs(self) -> None:
        for d in (self.data_root, self.model_root):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
