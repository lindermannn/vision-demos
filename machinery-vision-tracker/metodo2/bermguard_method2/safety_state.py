"""Safety state decision module for BermGuard Method 2.

Public API
----------
SafetyConfig         -- validated threshold/operational parameters
SafetyObservation    -- typed input contract for a single frame evaluation
SafetyDecision       -- typed output with traceable diagnostics
decide_safety()      -- pure function: (observation, config) -> decision

Decision rules (in priority order)
-----------------------------------
1. **Hazard observed** – If ``overlap_observed`` is True AND both
   ``berm_status`` and ``vehicle_status`` are ``'observed'``, emit RED.
   This rule bypasses the confidence gate intentionally: it is the
   conservative hazard-present rule.

2. **Staleness gate** – If ``distance_age_s`` exceeds ``distance_ttl_s``,
   emit UNKNOWN (stale measurement).

3. **Invalid calibration gate** – If ``calibration_status == 'invalid'``,
   emit UNKNOWN.

4. **Berm quality gate** – If ``berm_status != 'observed'`` (including
   ``'temporal_estimate'``, ``'historical_reference'``, ``'unknown'``),
   emit UNKNOWN.

5. **Vehicle quality gate** – If ``vehicle_status != 'observed'``,
   emit UNKNOWN.

6. **Confidence gate** – If ``confidence < min_confidence``, emit UNKNOWN.

7. **Metric decision** – Calibration is ``'calibrated'`` or ``'estimated'``
   and ``distance_m`` is available:
   * distance_m < red_threshold_m  -> RED
   * red_threshold_m <= distance_m < yellow_threshold_m  -> YELLOW
   * distance_m >= yellow_threshold_m -> GREEN

8. **Pixel-only estimated path** – Calibration is ``'pixel_only'``:
   main ``state`` = UNKNOWN (no metric basis);
   ``pixel_estimated_state`` populated from ``distance_px`` and labeled
   explicitly as an estimate.  MUST NOT be treated as metric.

9. **Unavailable / fallthrough** – emit UNKNOWN.

Prohibited by design
---------------------
* ``distance_m`` must not be set when calibration is not metric
  (enforced in ``SafetyObservation.__post_init__``).
* ``pixel_estimated_state`` is only populated for ``pixel_only``
  calibration; it is ``None`` in all other modes.
* YELLOW and GREEN are only reachable via the metric decision path (rule 7).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

SafetyLevel = Literal["red", "yellow", "green", "unknown"]

_VALID_CALIBRATION: frozenset[str] = frozenset(
    {"calibrated", "estimated", "pixel_only", "unavailable", "invalid"}
)
_VALID_OBSERVATION: frozenset[str] = frozenset(
    {"observed", "temporal_estimate", "historical_reference", "unknown"}
)
_METRIC_CALIBRATION: frozenset[str] = frozenset({"calibrated", "estimated"})


# ---------------------------------------------------------------------------
# SafetyConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SafetyConfig:
    """Thresholds and operational parameters for safety decisions.

    Invariants validated at construction:
    * ``0 < red_threshold_m < yellow_threshold_m``
      (semantic: RED threshold is strictly less than YELLOW threshold)
    * ``0 < red_threshold_px < yellow_threshold_px``
    * ``0.0 <= min_confidence <= 1.0``
    * ``distance_ttl_s > 0``

    Parameters
    ----------
    red_threshold_m:
        Metric distance in metres below which the safety state is RED.
        Requires metric calibration.
    yellow_threshold_m:
        Metric distance in metres below which the safety state is YELLOW
        (and above ``red_threshold_m``).  Requires metric calibration.
    red_threshold_px:
        Pixel distance below which the *pixel-only estimated* state is RED.
        Advisory only; never promoted to a metric result.
    yellow_threshold_px:
        Pixel distance below which the *pixel-only estimated* state is YELLOW.
        Advisory only; never promoted to a metric result.
    min_confidence:
        Minimum aggregate confidence required for a non-UNKNOWN metric state.
        The overlap (hazard) rule bypasses this gate intentionally.
    distance_ttl_s:
        Maximum age in seconds for a distance measurement before it is
        treated as stale and the state falls back to UNKNOWN.
    """

    red_threshold_m: float = 0.5
    yellow_threshold_m: float = 2.0
    red_threshold_px: float = 20.0
    yellow_threshold_px: float = 80.0
    min_confidence: float = 0.3
    distance_ttl_s: float = 0.5

    def __post_init__(self) -> None:
        errors: list[str] = []
        if not (self.red_threshold_m > 0):
            errors.append(
                f"red_threshold_m must be > 0, got {self.red_threshold_m}"
            )
        if not (self.yellow_threshold_m > self.red_threshold_m):
            errors.append(
                f"yellow_threshold_m ({self.yellow_threshold_m}) must be > "
                f"red_threshold_m ({self.red_threshold_m}); "
                "semantic constraint violated: red < yellow"
            )
        if not (self.red_threshold_px > 0):
            errors.append(
                f"red_threshold_px must be > 0, got {self.red_threshold_px}"
            )
        if not (self.yellow_threshold_px > self.red_threshold_px):
            errors.append(
                f"yellow_threshold_px ({self.yellow_threshold_px}) must be > "
                f"red_threshold_px ({self.red_threshold_px})"
            )
        if not (0.0 <= self.min_confidence <= 1.0):
            errors.append(
                f"min_confidence must be in [0, 1], got {self.min_confidence}"
            )
        if not (self.distance_ttl_s > 0):
            errors.append(
                f"distance_ttl_s must be > 0, got {self.distance_ttl_s}"
            )
        if errors:
            raise ValueError("; ".join(errors))


# ---------------------------------------------------------------------------
# SafetyObservation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SafetyObservation:
    """All inputs required for a single frame safety decision.

    This is a typed boundary contract.  Callers are responsible for:
    * Setting ``distance_m`` only when calibration is metric (enforced here).
    * Not passing ``distance_m`` derived from ``historical_reference`` or
      ``temporal_estimate`` data as if it were a current observation.
    * Setting ``distance_age_s = 0.0`` for same-frame measurements.
    * Setting ``overlap_observed = True`` only when geometry confirms actual
      pixel-space intersection of berm and vehicle masks in the current frame.

    Parameters
    ----------
    frame_index:
        Monotonic frame counter; used for correlation and staleness.
    timestamp:
        Seconds since epoch or stream start; used for staleness detection.
    distance_px:
        Pixel-space distance from vehicle ground point to berm toe.
        ``None`` if not computable.
    distance_m:
        Ground-plane metric distance (metres).  ``None`` if calibration is
        unavailable or non-metric.  Contract: MUST be ``None`` when
        ``calibration_status`` is not ``'calibrated'`` or ``'estimated'``.
    calibration_status:
        One of ``'calibrated'``, ``'estimated'``, ``'pixel_only'``,
        ``'unavailable'``, ``'invalid'``.
    observation_status:
        Provenance of the *distance measurement itself* (not the berm or
        vehicle independently).
    berm_status:
        Observation status of the berm geometry used to compute the distance.
    vehicle_status:
        Observation status of the vehicle track used to compute the distance.
    overlap_observed:
        ``True`` when current-frame geometry shows confirmed intersection of
        berm mask and vehicle mask.  The caller must guarantee both masks are
        from the current observed frame before setting this to ``True``.
    confidence:
        Aggregate confidence in [0, 1] for this observation.
    distance_age_s:
        Seconds elapsed since the distance was computed.  Use 0.0 for
        same-frame measurements.  Must be >= 0.
    diagnostics:
        Pass-through diagnostic codes from upstream modules.
    """

    frame_index: int
    timestamp: float
    distance_px: Optional[float]
    distance_m: Optional[float]
    calibration_status: str
    observation_status: str
    berm_status: str
    vehicle_status: str
    overlap_observed: bool
    confidence: float
    distance_age_s: float = 0.0
    diagnostics: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        errors: list[str] = []
        if self.calibration_status not in _VALID_CALIBRATION:
            errors.append(
                f"calibration_status '{self.calibration_status}' is not valid; "
                f"expected one of {sorted(_VALID_CALIBRATION)}"
            )
        if self.observation_status not in _VALID_OBSERVATION:
            errors.append(
                f"observation_status '{self.observation_status}' is not valid; "
                f"expected one of {sorted(_VALID_OBSERVATION)}"
            )
        if self.berm_status not in _VALID_OBSERVATION:
            errors.append(
                f"berm_status '{self.berm_status}' is not valid; "
                f"expected one of {sorted(_VALID_OBSERVATION)}"
            )
        if self.vehicle_status not in _VALID_OBSERVATION:
            errors.append(
                f"vehicle_status '{self.vehicle_status}' is not valid; "
                f"expected one of {sorted(_VALID_OBSERVATION)}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            errors.append(
                f"confidence must be in [0, 1], got {self.confidence}"
            )
        if self.distance_age_s < 0.0:
            errors.append(
                f"distance_age_s must be >= 0, got {self.distance_age_s}"
            )
        if (
            self.distance_m is not None
            and self.calibration_status not in _METRIC_CALIBRATION
        ):
            errors.append(
                f"distance_m is set ({self.distance_m}) but calibration_status "
                f"is '{self.calibration_status}'; metric distance requires "
                f"calibration_status in {sorted(_METRIC_CALIBRATION)}.  "
                "Do not promote historical_reference or temporal_estimate "
                "distances as if they were current metric observations."
            )
        if errors:
            raise ValueError("; ".join(errors))


# ---------------------------------------------------------------------------
# SafetyDecision
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SafetyDecision:
    """Result of a safety evaluation for a single frame.

    Attributes
    ----------
    state:
        The authoritative safety level.

        * ``'red'``: reachable via observed overlap (hazard rule) or metric
          distance below ``red_threshold_m`` (metric rule).
        * ``'yellow'``: reachable ONLY via metric rule (calibration required).
        * ``'green'``: reachable ONLY via metric rule (calibration required).
        * ``'unknown'``: all other cases; see ``reason`` and ``diagnostics``.

    pixel_estimated_state:
        Populated ONLY when ``calibration_status == 'pixel_only'``.
        Derived from ``distance_px`` against ``red_threshold_px`` /
        ``yellow_threshold_px``.  This is an ADVISORY ESTIMATE and MUST NOT
        be treated as a metric measurement or presented as one.
        Is ``None`` in all other calibration modes.

    confidence:
        Propagated from the observation when state is ``'red'``, ``'yellow'``,
        or ``'green'``.  Always ``0.0`` when state is ``'unknown'``.

    reason:
        Human-readable single-sentence justification of the decision.

    diagnostics:
        Ordered tuple of traceable diagnostic codes from this module and
        upstream pass-through codes.  Use for logging and audit trails.

    frame_index:
        Echoed from the input observation for correlation.

    timestamp:
        Echoed from the input observation for correlation.
    """

    state: SafetyLevel
    pixel_estimated_state: Optional[SafetyLevel]
    confidence: float
    reason: str
    diagnostics: Tuple[str, ...]
    frame_index: int
    timestamp: float


# ---------------------------------------------------------------------------
# Decision engine
# ---------------------------------------------------------------------------

def decide_safety(
    obs: SafetyObservation,
    config: SafetyConfig,
) -> SafetyDecision:
    """Evaluate the safety state for a single observation frame.

    Parameters
    ----------
    obs:
        Typed observation for the current frame.
    config:
        Validated safety thresholds and operational parameters.

    Returns
    -------
    SafetyDecision
        The authoritative safety state with full diagnostic trace.

    Notes
    -----
    This function is pure (no side effects, no global state).
    All decision rationale is captured in the returned ``diagnostics`` tuple.
    """
    diag: list[str] = list(obs.diagnostics)

    # ------------------------------------------------------------------
    # Rule 1 – Hazard observed: currently observed geometric overlap.
    # This is the conservative hazard rule.  It requires BOTH berm and
    # vehicle to be currently 'observed', but bypasses the confidence
    # gate because a geometric overlap is a first-class safety signal.
    # ------------------------------------------------------------------
    if obs.overlap_observed:
        if (
            obs.berm_status == "observed"
            and obs.vehicle_status == "observed"
        ):
            diag.append("hazard_rule:overlap_observed_both_confirmed_observed")
            return SafetyDecision(
                state="red",
                pixel_estimated_state=None,
                confidence=obs.confidence,
                reason=(
                    "Current-frame geometric overlap between vehicle and berm "
                    "confirmed; both berm and vehicle have 'observed' status."
                ),
                diagnostics=tuple(diag),
                frame_index=obs.frame_index,
                timestamp=obs.timestamp,
            )
        # Overlap claimed but at least one party is not currently observed.
        diag.append(
            f"overlap_claimed_but_not_defensible:"
            f"berm={obs.berm_status},vehicle={obs.vehicle_status}"
        )

    # ------------------------------------------------------------------
    # Rule 2 – Staleness gate.
    # ------------------------------------------------------------------
    if obs.distance_age_s > config.distance_ttl_s:
        diag.append(
            f"staleness_gate:distance_age={obs.distance_age_s:.4f}s"
            f">ttl={config.distance_ttl_s:.4f}s"
        )
        return _unknown(
            obs,
            diag,
            f"Distance measurement age {obs.distance_age_s:.3f} s exceeds "
            f"TTL {config.distance_ttl_s:.3f} s; result is stale.",
        )

    # ------------------------------------------------------------------
    # Rule 3 – Invalid calibration gate.
    # ------------------------------------------------------------------
    if obs.calibration_status == "invalid":
        diag.append("calibration_gate:status_invalid")
        return _unknown(
            obs, diag, "Calibration status is 'invalid'; cannot make a decision."
        )

    # ------------------------------------------------------------------
    # Rule 4 – Berm observation quality gate.
    # ------------------------------------------------------------------
    if obs.berm_status != "observed":
        diag.append(f"berm_gate:status={obs.berm_status}_not_observed")
        return _unknown(
            obs,
            diag,
            f"Berm observation status is '{obs.berm_status}', not 'observed'; "
            "historical, estimated, or unknown berm geometry cannot support "
            "a metric safety decision.",
        )

    # ------------------------------------------------------------------
    # Rule 5 – Vehicle observation quality gate.
    # ------------------------------------------------------------------
    if obs.vehicle_status != "observed":
        diag.append(f"vehicle_gate:status={obs.vehicle_status}_not_observed")
        return _unknown(
            obs,
            diag,
            f"Vehicle observation status is '{obs.vehicle_status}', not "
            "'observed'; temporal or unknown vehicle position cannot support "
            "a metric safety decision.",
        )

    # ------------------------------------------------------------------
    # Rule 6 – Confidence gate.
    # ------------------------------------------------------------------
    if obs.confidence < config.min_confidence:
        diag.append(
            f"confidence_gate:{obs.confidence:.4f}<min={config.min_confidence:.4f}"
        )
        return _unknown(
            obs,
            diag,
            f"Aggregate confidence {obs.confidence:.3f} is below the minimum "
            f"required {config.min_confidence:.3f}.",
        )

    # ------------------------------------------------------------------
    # Rule 7 – Metric decision path.
    # ------------------------------------------------------------------
    if obs.calibration_status in _METRIC_CALIBRATION:
        if obs.distance_m is None:
            diag.append(
                f"metric_gate:calibration={obs.calibration_status}"
                "_but_distance_m_is_none"
            )
            return _unknown(
                obs,
                diag,
                f"Calibration is '{obs.calibration_status}' but distance_m "
                "is None; cannot compute metric safety state.",
            )
        d = obs.distance_m
        if d < config.red_threshold_m:
            diag.append(
                f"metric_decision:distance_m={d:.6f}"
                f"<red_threshold={config.red_threshold_m:.6f}"
            )
            return SafetyDecision(
                state="red",
                pixel_estimated_state=None,
                confidence=obs.confidence,
                reason=(
                    f"Metric distance {d:.3f} m is below RED threshold "
                    f"{config.red_threshold_m:.3f} m "
                    f"(calibration: {obs.calibration_status})."
                ),
                diagnostics=tuple(diag),
                frame_index=obs.frame_index,
                timestamp=obs.timestamp,
            )
        if d < config.yellow_threshold_m:
            diag.append(
                f"metric_decision:distance_m={d:.6f}"
                f">=red={config.red_threshold_m:.6f}"
                f"<yellow={config.yellow_threshold_m:.6f}"
            )
            return SafetyDecision(
                state="yellow",
                pixel_estimated_state=None,
                confidence=obs.confidence,
                reason=(
                    f"Metric distance {d:.3f} m is between RED threshold "
                    f"{config.red_threshold_m:.3f} m and YELLOW threshold "
                    f"{config.yellow_threshold_m:.3f} m "
                    f"(calibration: {obs.calibration_status})."
                ),
                diagnostics=tuple(diag),
                frame_index=obs.frame_index,
                timestamp=obs.timestamp,
            )
        diag.append(
            f"metric_decision:distance_m={d:.6f}"
            f">=yellow_threshold={config.yellow_threshold_m:.6f}"
        )
        return SafetyDecision(
            state="green",
            pixel_estimated_state=None,
            confidence=obs.confidence,
            reason=(
                f"Metric distance {d:.3f} m is at or above YELLOW threshold "
                f"{config.yellow_threshold_m:.3f} m "
                f"(calibration: {obs.calibration_status})."
            ),
            diagnostics=tuple(diag),
            frame_index=obs.frame_index,
            timestamp=obs.timestamp,
        )

    # ------------------------------------------------------------------
    # Rule 8 – Pixel-only estimated path.
    # main state = UNKNOWN (no metric basis); pixel_estimated_state is
    # advisory only and MUST NOT be treated as a metric measurement.
    # ------------------------------------------------------------------
    if obs.calibration_status == "pixel_only":
        pixel_est = _pixel_estimated_state(obs.distance_px, config)
        diag.append(f"pixel_only:estimated_state={pixel_est}")
        if obs.distance_px is not None:
            diag.append(f"pixel_only:distance_px={obs.distance_px:.4f}")
        else:
            diag.append("pixel_only:distance_px=none")
        return SafetyDecision(
            state="unknown",
            pixel_estimated_state=pixel_est,
            confidence=0.0,
            reason=(
                "No metric calibration available (pixel_only mode). "
                "The main safety state is UNKNOWN. "
                "pixel_estimated_state is advisory only and must NOT be "
                "treated as or presented as a metric measurement."
            ),
            diagnostics=tuple(diag),
            frame_index=obs.frame_index,
            timestamp=obs.timestamp,
        )

    # ------------------------------------------------------------------
    # Rule 9 – Fallthrough (unavailable or unrecognised calibration).
    # ------------------------------------------------------------------
    diag.append(
        f"fallthrough:calibration_status={obs.calibration_status}_unsupported"
    )
    return _unknown(
        obs,
        diag,
        f"Calibration status '{obs.calibration_status}' does not provide "
        "sufficient information for a safety decision.",
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pixel_estimated_state(
    distance_px: Optional[float],
    config: SafetyConfig,
) -> SafetyLevel:
    """Return an ESTIMATED pixel-only safety level (advisory, not metric).

    Returns ``'unknown'`` when ``distance_px`` is None.
    """
    if distance_px is None:
        return "unknown"
    if distance_px < config.red_threshold_px:
        return "red"
    if distance_px < config.yellow_threshold_px:
        return "yellow"
    return "green"


def _unknown(
    obs: SafetyObservation,
    diag: list[str],
    reason: str,
) -> SafetyDecision:
    """Build an UNKNOWN decision with zero confidence."""
    return SafetyDecision(
        state="unknown",
        pixel_estimated_state=None,
        confidence=0.0,
        reason=reason,
        diagnostics=tuple(diag),
        frame_index=obs.frame_index,
        timestamp=obs.timestamp,
    )
