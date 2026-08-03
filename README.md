# Driver-State & Adaptive Cockpit

Estimates a driver's **fatigue state** from multimodal driving + physiological
signals using **sequence models (LSTM / Transformer)**, then **adapts ADAS and
cockpit settings** in real time through a transparent policy engine — served over
FastAPI with a live dashboard.

> **Why this exists.** Human-centric ADAS is where Alfa Romeo, Lancia and Ferrari
> are investing: **Alfa's Driver Attention Assist** infers fatigue from lateral
> movement, steering and time-on-task; **Lancia's SALA cockpit** adapts light,
> sound and interface to the driver; **Ferrari's psychophysical-monitoring &
> coaching** patents read driver state on track. This project reproduces that
> loop end-to-end: **sense → estimate driver state → adapt the car**.

---

## What it does

```
 multimodal signals            sequence model              policy engine
 (behavioural + physio) ──▶  LSTM / Transformer  ──▶  ADAS + cockpit adaptation ──▶ FastAPI + dashboard
  60-second window             fatigue state                (SALA / Attention-Assist
  @ 1 Hz                       + calibrated score            / Ferrari-style)
```

- **Signals (16 channels):** lateral deviation, steering-reversal rate, lane-keeping
  variability, time-headway, throttle/brake jerk; recent hard-brakes / near-misses;
  time-on-task, time-since-break, night; heart rate, HRV, blink rate, **PERCLOS**
  (eye-closure), gaze-off-road.
- **Model:** predicts fatigue state (`alert` / `drowsy` / `critical`) from a 60 s
  window. No single timestep is reliable — the LSTM/Transformer integrate the
  window over time.
- **Policy:** maps the state to concrete, bounded, **explainable** adaptations —
  earlier collision warnings, firmer lane-keeping, larger following gaps, plus
  cockpit changes (alert tone/volume, display brightness, ambient light, cabin
  cooling, break prompts).

---

## Results

Sequence models over 60 s windows, split **by driver** (no driver in both train
and test). Reproducible via `python -m driver_state.cli train`.

| Model | Accuracy | Macro-F1 | ECE (calibration) |
|---|---|---|---|
| **LSTM** (serving) | **0.972** | **0.971** | **0.003** |
| Transformer | 0.971 | 0.970 | 0.004 |

<sub>80 drivers × 3 sessions, split by driver; 3-class fatigue over 60 s windows.
The better model (LSTM here) is auto-selected for serving.</sub>

Well below 0.02 ECE means the model is **calibrated** — its "70% drowsy" is right
~70% of the time, which matters when the score gates a real warning. Signal
**drift** (PSI/KS between alert and fatigued cohorts) is reported via `/metrics`.

---

## The live dashboard

`GET /` serves a single-page React dashboard (loaded via CDN, no build step):

- **driver-state timeline** — predicted fatigue vs. ground truth over a session,
- a **live driver-state badge** (alert / drowsy / critical) with confidence,
- the **adaptive cockpit panel** — ADAS + cabin settings that change with the state,
- **model comparison + calibration + drift** metrics.

Replays synthetic sessions so it works with zero configuration.

---

## API

| Endpoint | Description |
|---|---|
| `GET /` | live dashboard |
| `GET /health` | liveness + serving model |
| `GET /schema` | feature-column order for a window |
| `POST /driver_state` | `{window:[[...]]}` (60×16) → fatigue-state estimate |
| `POST /policy` | window (+`is_night`,`time_on_task_min`,`stress`) → ADAS + cockpit policy |
| `GET /session/simulate?index=` | replay a demo session with per-window predictions |
| `GET /metrics` | model comparison, calibration, drift |

Example — a window flagged as drowsy yields more assertive assistance:

```jsonc
POST /policy
{ "driver_state": { "fatigue_state": "drowsy", "fatigue_score": 0.58, ... },
  "policy": {
    "fcw_lead_time_s": 3.0, "following_distance_s": 2.2, "lane_keep_strictness": "firm",
    "alert_tone": "standard", "display_brightness": 0.75, "ambient_light": "neutral",
    "cabin_cooling_nudge": true, "break_recommendation": false,
    "rationale": ["fatigue state 'drowsy' (severity 0.58)", "..."] } }
```

---

## Quickstart

Requires Python 3.10–3.12.

```bash
python -m venv .venv           # on Windows, see the note below
# activate, then:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python -m driver_state.cli train          # generate + train both models (~a few min, CPU)
uvicorn driver_state.api.main:app --reload   # http://localhost:8000
```

> **Windows note.** PyTorch bundles very deeply-nested files that can exceed the
> 260-char path limit under a long project path. Either enable Win32 long paths,
> or create the venv at a short path (e.g. `python -m venv C:\dsc-venv`). Linux /
> Docker are unaffected.

### Docker (self-contained: trains at build, then serves)
```bash
docker compose up --build     # http://localhost:8000
```

### Tests
```bash
pytest -q                     # simulator, dataset, model, policy, API
```
CI runs the suite on every push (`.github/workflows/ci.yml`).

---

## Project layout

```
driver_state/
  config.py                 env-driven settings
  signals.py                the 16-channel signal schema
  generation/simulator.py   latent fatigue/stress -> multimodal signals
  data/windows.py           sliding windows + by-driver split + scaling
  models/
    lstm.py, transformer.py sequence classifiers
    train.py                train both, select by macro-F1, eval + calibration
    infer.py                serving wrapper (window -> state)
  policy/engine.py          driver state -> ADAS + cockpit policy
  monitoring/metrics.py     calibration (ECE) + drift (PSI/KS)
  api/                      FastAPI app + schemas + serving layer
  cli.py                    generate -> train -> eval
dashboard/index.html        single-file React + Chart.js dashboard
tests/                      pytest suite
Dockerfile · docker-compose.yml · .github/workflows/ci.yml
```

## Cloud-ready notes
- **Training** maps to SageMaker / an EC2 GPU job; the pipeline is a single entrypoint.
- **Serving** (FastAPI + a small model) runs on ECS Fargate / any container host.
  Note PyTorch's memory footprint — a ~1 GB instance is comfortable (a 512 MB
  free tier is tight for torch).
- **Monitoring:** the in-process calibration/drift metrics would feed an
  EvidentlyAI or CloudWatch dashboard in production.

## Roadmap
- Real signals via CARLA or a public drowsy-driving dataset (adapter alongside the simulator).
- Per-driver personalisation (adaptation baselines from a driver's own history).
- On-device quantised model for the edge (cabin ECU).

## Context
Inspired by Alfa Romeo Driver Attention Assist, Lancia's Level-2 ADAS + SALA
interface, and Ferrari's psychophysical driver-monitoring / coaching work. Data
is fully synthetic; no real driver data is used.
