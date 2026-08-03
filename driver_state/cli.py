"""
Pipeline CLI.

    python -m driver_state.cli train    # generate -> train both models -> eval (default)
    python -m driver_state.cli demo     # save demo sessions for the dashboard

Usage of `train` also saves demo sessions so the API/dashboard work immediately.
"""
from __future__ import annotations

import argparse
import time

from .config import settings
from .generation import save_demo_sessions
from .models.train import train_all


def cmd_train():
    t = time.time()
    train_all(settings)
    save_demo_sessions(settings)
    print(f"Done in {time.time()-t:.1f}s. Artifacts in {settings.model_root}")


def cmd_demo():
    save_demo_sessions(settings)
    print(f"Saved demo sessions to {settings.data_root}")


COMMANDS = {"train": cmd_train, "demo": cmd_demo}


def main():
    ap = argparse.ArgumentParser(description="Driver-state pipeline")
    ap.add_argument("command", nargs="?", default="train", choices=list(COMMANDS))
    COMMANDS[ap.parse_args().command]()


if __name__ == "__main__":
    main()
