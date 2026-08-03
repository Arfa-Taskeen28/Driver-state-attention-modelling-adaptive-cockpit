"""
Export a static version of the dashboard for GitHub Pages (no backend).

Precomputes each demo session's per-window predictions + the metrics report to
JSON, and copies the dashboard into docs/. The dashboard falls back to these
files when no live API is reachable, so the Pages site is fully interactive
(session replay, cockpit, metrics) without a server.

    python -m scripts.export_static_demo
"""
from __future__ import annotations

import json
import pathlib
import shutil

from driver_state.api.service import DriverStateService
from driver_state.config import REPO_ROOT, settings

DOCS = REPO_ROOT / "docs"
DATA = DOCS / "data"


def main():
    svc = DriverStateService(settings)
    DATA.mkdir(parents=True, exist_ok=True)
    n = svc.n_demo()
    for i in range(n):
        (DATA / f"session_{i}.json").write_text(json.dumps(svc.simulate_session(i)))
    (DATA / "metrics.json").write_text(json.dumps(svc.metrics_report()))
    (DATA / "manifest.json").write_text(json.dumps({"count": n}))
    shutil.copy(REPO_ROOT / "dashboard" / "index.html", DOCS / "index.html")
    print(f"exported {n} sessions + metrics to {DATA}")


if __name__ == "__main__":
    main()
