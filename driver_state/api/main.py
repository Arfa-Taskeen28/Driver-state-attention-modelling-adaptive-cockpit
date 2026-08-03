"""
FastAPI service for driver-state estimation + adaptive cockpit.

GET  /                    the live dashboard (static React page)
GET  /health              liveness + serving model
GET  /schema              feature column order for a window
POST /driver_state        window of signals -> fatigue-state estimate
POST /policy              window (+context) -> ADAS + cockpit policy
GET  /session/simulate    replay a demo session with per-window predictions
GET  /metrics             model comparison, calibration, drift
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse

from .. import __version__
from ..config import REPO_ROOT
from ..signals import FEATURE_COLUMNS
from .schemas import (DriverStateResponse, HealthResponse, PolicyRequest,
                      PolicyResponse, SchemaResponse, WindowRequest)
from .service import get_service

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")

app = FastAPI(
    title="Driver-State & Adaptive Cockpit API",
    version=__version__,
    description="Estimates driver fatigue from multimodal signals with sequence "
                "models and adapts ADAS + cockpit settings.",
)

DASHBOARD = REPO_ROOT / "dashboard" / "index.html"


@app.get("/", include_in_schema=False)
def dashboard():
    if DASHBOARD.exists():
        return FileResponse(DASHBOARD)
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health():
    svc = get_service()
    return HealthResponse(status="ok", serving_model=svc.model.name,
                          n_features=len(FEATURE_COLUMNS),
                          window_len=svc.cfg.window_len)


@app.get("/schema", response_model=SchemaResponse)
def schema():
    svc = get_service()
    return SchemaResponse(feature_columns=FEATURE_COLUMNS,
                          window_len=svc.cfg.window_len)


@app.post("/driver_state", response_model=DriverStateResponse)
def driver_state(req: WindowRequest):
    try:
        return get_service().predict(req.window)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/policy", response_model=PolicyResponse)
def policy(req: PolicyRequest):
    try:
        return get_service().policy(req.window, is_night=req.is_night,
                                    time_on_task_min=req.time_on_task_min,
                                    stress=req.stress)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/session/simulate")
def session_simulate(index: int = Query(0, ge=0)):
    return get_service().simulate_session(index)


@app.get("/metrics")
def metrics():
    return get_service().metrics_report()
