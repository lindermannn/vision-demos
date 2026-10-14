"""Default settings for the push-up demo.

The tuning below is adapted from the pull-up profile for push-up detection.
Change it here, or pass another profile file to main.py.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "data" / "output"
CACHE_DIR = ROOT / "data" / "cache"
CALIBRATION_DIR = ROOT / "data" / "calibrations"

DEFAULT_TUNING = {
    "minimum_landmark_score": 0.35,
    "minimum_visible_fraction": 0.7,
    "minimum_rom_degrees": 30.0,
    "flexed_angle_degrees": 105.0,
    "minimum_duration_s": 0.35,
    "maximum_duration_s": 8.0,
    "stable_rest_s": 0.1,
    "rest_score": 0.25,
    "movement_score": 0.42,
    "peak_score": 0.7,
    "head_motion_weight": 0.0,  # Push-ups don't use head motion
    "head_fallback_enabled": False,
    "head_visibility_score": 0.55,
    "head_cycle_coverage": 0.8,
    "arm_cycle_coverage_trigger": 0.7,
    "head_rise_fraction": 0.04,
    "shoulder_rise_fraction": 0.025,
    "peak_spacing_s": 0.35,
    "local_peak_window_s": 0.5,
    "long_cycle_split_s": 2.0,
    "camera_stable_confirmed": False,
}

LAYOUTS = ("reel", "horizontal")