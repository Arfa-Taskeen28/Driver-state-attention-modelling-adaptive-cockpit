"""
driver_state — Driver-state & attention modelling + adaptive cockpit.

Estimates driver fatigue/stress from multimodal driving + physiological signals
with sequence models (LSTM / Transformer), then adapts ADAS aggressiveness and
cockpit settings via a transparent policy engine — served over FastAPI with a
live dashboard.

Inspired by Alfa Romeo's Driver Attention Assist, Lancia's Level-2 ADAS + SALA
cockpit, and Ferrari's psychophysical driver-monitoring / coaching work.
"""

__version__ = "0.1.0"
