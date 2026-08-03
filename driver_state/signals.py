"""
Signal schema — the multimodal channels the model consumes.

Grouped by modality so the README / dashboard can explain provenance. Alfa's
Driver Attention Assist infers fatigue from *behavioural* signals (lateral
movement, steering, time-on-task); physiological channels (Ferrari-style
psychophysical monitoring) are optional but improve early detection.
"""
from __future__ import annotations

# Behavioural / vehicle-derived
BEHAVIOURAL = [
    "lateral_deviation_m",     # drift within lane
    "steering_reversal_rate",  # micro-corrections per minute
    "lane_offset_std",         # lane-keeping variability
    "time_headway_s",          # gap to lead vehicle
    "speed_kph",
    "throttle_brake_jerk",     # longitudinal control smoothness
]

# Event history (trailing activity)
EVENT = [
    "recent_hard_brakes",      # count in trailing minute
    "recent_near_misses",
]

# Context / session
CONTEXT = [
    "time_on_task_min",
    "time_since_break_min",
    "is_night",
]

# Physiological (optional; synthetic here)
PHYSIOLOGICAL = [
    "heart_rate_bpm",
    "hrv_sdnn_ms",
    "blink_rate_per_min",
    "perclos",                 # fraction of time eyes closed
    "gaze_off_road_ratio",
]

FEATURE_COLUMNS = BEHAVIOURAL + EVENT + CONTEXT + PHYSIOLOGICAL
N_FEATURES = len(FEATURE_COLUMNS)

# Fatigue state classes
CLASS_NAMES = ["alert", "drowsy", "critical"]
N_CLASSES = len(CLASS_NAMES)
