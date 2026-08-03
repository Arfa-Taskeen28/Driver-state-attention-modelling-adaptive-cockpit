"""
Adaptive cockpit + ADAS policy engine.

Maps an estimated driver state to concrete ADAS and cockpit adaptations. The
policy is transparent, bounded and monotonic (more fatigue -> earlier warnings,
firmer lane-keeping, more stimulating cabin, and eventually a break prompt) and
returns a rationale for every change — because driver-facing safety behaviour
must be explainable.

Inspired by Alfa's Driver Attention Assist (act on inferred fatigue), Lancia's
SALA cockpit (adaptive light/sound/interface), and Ferrari's driver-coaching.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class CockpitPolicy:
    # driver state that produced this policy
    fatigue_state: str
    fatigue_score: float
    # ADAS adaptations
    fcw_lead_time_s: float
    following_distance_s: float
    lane_keep_strictness: str          # standard | firm | max
    speed_limiter_advice: str | None
    # cockpit adaptations (SALA-style)
    alert_tone: str                    # soft | standard | urgent
    alert_volume: float                # 0..1
    display_brightness: float          # 0..1
    ambient_light: str                 # calm | neutral | energising
    suggestion_frequency: str          # low | normal | high
    cabin_cooling_nudge: bool
    break_recommendation: bool
    message: str
    rationale: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


def decide_policy(fatigue_state: str, fatigue_score: float,
                  stress: float | None = None,
                  time_on_task_min: float | None = None,
                  is_night: bool = False) -> CockpitPolicy:
    r = float(max(0.0, min(1.0, fatigue_score)))
    rationale = [f"fatigue state '{fatigue_state}' (severity {r:.2f})"]

    # --- ADAS: intervene earlier and hold larger margins as fatigue rises ----
    fcw = round(2.0 + 1.8 * r, 2)
    follow = round(1.6 + 1.0 * r, 2)
    lane = "max" if r >= 0.66 else "firm" if r >= 0.33 else "standard"
    rationale.append(f"forward-collision warning at {fcw}s, following gap {follow}s")
    rationale.append(f"lane-keeping set to '{lane}'")

    speed_advice = None
    if r >= 0.66:
        speed_advice = "reduce speed / hand over to ACC — critical fatigue"
        rationale.append("advised speed reduction")

    # --- Cockpit (SALA-style): stimulate a drowsy driver, stay calm otherwise -
    if r >= 0.66:
        tone, vol, bright, ambient, freq = "urgent", 0.9, 0.95, "energising", "high"
        cool = True
    elif r >= 0.33:
        tone, vol, bright, ambient, freq = "standard", 0.6, 0.75, "neutral", "normal"
        cool = True
    else:
        tone, vol, bright, ambient, freq = "soft", 0.35, 0.55, "calm", "low"
        cool = False
    if is_night:
        bright = round(min(bright, 0.7), 2)      # avoid glare at night
        rationale.append("brightness capped for night driving")

    # a stressed driver gets a calmer, less intrusive cabin
    if stress is not None and stress >= 0.6:
        tone = "soft" if tone != "urgent" else tone
        ambient = "calm" if r < 0.66 else ambient
        rationale.append(f"stress elevated ({stress:.2f}) — softening non-critical alerts")

    break_reco = r >= 0.66 or (time_on_task_min is not None and time_on_task_min >= 120)
    if break_reco:
        rationale.append("break recommended")
    message = ("Take a break soon — signs of drowsiness detected."
               if break_reco else
               "Attention nominal — assistance tuned to your state."
               if r < 0.33 else
               "Mild fatigue — assistance sensitivity increased.")

    return CockpitPolicy(
        fatigue_state=fatigue_state, fatigue_score=round(r, 4),
        fcw_lead_time_s=fcw, following_distance_s=follow, lane_keep_strictness=lane,
        speed_limiter_advice=speed_advice,
        alert_tone=tone, alert_volume=vol, display_brightness=round(bright, 2),
        ambient_light=ambient, suggestion_frequency=freq, cabin_cooling_nudge=cool,
        break_recommendation=break_reco, message=message, rationale=rationale)
