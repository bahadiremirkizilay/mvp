"""
Physiological Stress Estimator
================================
Estimates a stress level and a normalised stress score from HRV features
derived by the rPPG pipeline.

Design
------
The estimator is implemented as a class with a single public method
``estimate()``.  This keeps the module swappable: a future ML model only
needs to implement the same interface (same input dict, same output dict).

Rule-based model (v1)
---------------------
Inputs used:
    heart_rate (BPM)   — elevated HR is associated with stress / arousal
    rmssd (ms)         — lower RMSSD → reduced vagal tone → higher stress
    sdnn  (ms)         — lower SDNN  → reduced overall HRV → higher stress
    lf_hf             — higher LF/HF ratio → sympathetic dominance → stress

Each feature is mapped to a partial stress score in [0, 1] via a piecewise
linear (clip-normalise) function, then the four partial scores are combined
as a weighted average:

    weights: rmssd 0.40 | sdnn 0.25 | hr 0.20 | lf_hf 0.15

The final numeric score drives a three-level label:
    score ≥ 0.60  → HIGH
    score ≥ 0.35  → MEDIUM
    score <  0.35 → LOW

Reference ranges (literature / clinical guidelines):
    Resting HR:   60–80 BPM typical; > 100 BPM indicates stress/tachycardia
    RMSSD:        20–80 ms healthy resting range
    SDNN:         30–100 ms typical short-window healthy range
    LF/HF:        0.5–2.0 balanced; > 2 sympathetic dominance
"""

from typing import Dict, Any


# ---------------------------------------------------------------------------
# Feature normalisation helpers
# ---------------------------------------------------------------------------

def _clip_norm(value: float, low: float, high: float) -> float:
    """
    Map ``value`` to [0, 1] linearly between ``low`` (→ 0) and ``high`` (→ 1),
    clipping outside that range.
    """
    if high <= low:
        return 0.0
    return float(max(0.0, min(1.0, (value - low) / (high - low))))


# ---------------------------------------------------------------------------
# Public estimator class
# ---------------------------------------------------------------------------

class StressEstimator:
    """
    Rule-based physiological stress estimator.

    Instantiate once; call ``estimate()`` every processing cycle.

    Swapping to an ML model later only requires replacing this class (or
    sub-classing and overriding ``estimate()``) — the rest of the pipeline
    is unchanged.

    Parameters
    ----------
    weights : dict, optional
        Override feature weights.  Keys: ``rmssd``, ``sdnn``, ``hr``, ``lf_hf``.
        Values must sum to 1.0.  If None, the defaults are used.
    """

    _DEFAULT_WEIGHTS = {
        "rmssd": 0.40,
        "sdnn":  0.25,
        "hr":    0.20,
        "lf_hf": 0.15,
    }

    def __init__(self, weights: Dict[str, float] = None) -> None:
        self.weights = weights or dict(self._DEFAULT_WEIGHTS)

    # ------------------------------------------------------------------

    def estimate(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate stress from HRV features.

        Parameters
        ----------
        features : dict
            Expected keys (all float, missing keys default to 0.0):
                ``heart_rate``  — smoothed BPM
                ``rmssd``       — RMSSD in ms
                ``sdnn``        — SDNN in ms
                ``lf_hf``       — LF/HF ratio

        Returns
        -------
        dict
            ``stress_score`` : float in [0, 1]
                0 = no stress, 1 = maximum stress.
            ``stress_level`` : str
                "LOW", "MEDIUM", or "HIGH".
            ``partial_scores`` : dict
                Per-feature partial scores for introspection / debugging.
        """
        hr    = float(features.get("heart_rate", 0.0))
        rmssd = float(features.get("rmssd",      0.0))
        sdnn  = float(features.get("sdnn",        0.0))
        lf_hf = float(features.get("lf_hf",       0.0))

        # ---- Partial scores (higher = more stress) ----

        # RMSSD: 20 ms (high stress) → 80 ms (low stress) — inverted
        s_rmssd = 1.0 - _clip_norm(rmssd, 20.0, 80.0) if rmssd > 0 else 0.5

        # SDNN: 30 ms (high stress) → 100 ms (low stress) — inverted
        s_sdnn  = 1.0 - _clip_norm(sdnn,  30.0, 100.0) if sdnn  > 0 else 0.5

        # Heart rate: 60 BPM (relaxed) → 100 BPM (stressed)
        s_hr    = _clip_norm(hr,    60.0, 100.0) if hr    > 0 else 0.5

        # LF/HF: 0.5 (parasympathetic) → 3.0 (sympathetic dominance)
        s_lf_hf = _clip_norm(lf_hf, 0.5,   3.0) if lf_hf > 0 else 0.5

        partial = {
            "rmssd": s_rmssd,
            "sdnn":  s_sdnn,
            "hr":    s_hr,
            "lf_hf": s_lf_hf,
        }

        # ---- Weighted combination ----
        score = sum(self.weights[k] * partial[k] for k in self.weights)
        score = float(max(0.0, min(1.0, score)))

        # ---- Label ----
        if score >= 0.60:
            level = "HIGH"
        elif score >= 0.35:
            level = "MEDIUM"
        else:
            level = "LOW"

        return {
            "stress_score":   score,
            "stress_level":   level,
            "partial_scores": partial,
        }
