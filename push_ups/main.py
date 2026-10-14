"""Analyze one push-up clip and write the results under data/output/.

Usage:
    python main.py clips/set-of-20.mp4
    python main.py clips/set-of-20.mp4 --calibration data/calibrations/my-pushups.json
    python main.py clips/set-of-20.mp4 --layout horizontal --static-camera

Each run creates a folder with the analysis video, the rep data (JSON), a text
report and a record of the run. Calibration is optional: without it, reps are
counted and timed, but not graded.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
from pathlib import Path

from config import CACHE_DIR, DEFAULT_TUNING, LAYOUTS, OUTPUT_DIR
from pushup_demo.extract import cached_track, file_sha256
from pushup_demo.pose_contract import load_pose_track
from pushup_demo.pushup_calibration import GRADES, apply_calibration
from pushup_demo.pushup_profiles import PushupTuning, analyze_pushup_profile, height_curve
from pushup_demo.render import render


def clears_floor(height: float | None) -> bool | None:
    """Chest at or near the floor. Estimated from pose, not verified."""
    if height is None or not math.isfinite(height):
        return None
    return height >= 0


def report_text(clip: Path, events: list[dict], calibration: dict | None, tuning: PushupTuning) -> str:
    lines = [
        f"Clip: {clip.name}",
        f"Reps counted: {len(events)}",
        f"Camera confirmed static: {'yes' if tuning.camera_stable_confirmed else 'no'}",
    ]
    if events:
        durations = [e["end"] - e["start"] for e in events]
        lines.append(f"Set duration: {events[-1]['end'] - events[0]['start']:.1f} s")
        lines.append(f"Median rep duration: {statistics.median(durations):.2f} s")
    if calibration:
        lines.append(f"Calibration: {calibration['name']} (grades: " + ", ".join(
            f"{GRADES[key]} from {calibration['thresholds'][key]:+.2f}"
            for key in ("rom_full_alto", "rom_semi_alto", "rom_full")) + ")")
        counts: dict[str, int] = {}
        for event in events:
            counts[event["grade_label"]] = counts.get(event["grade_label"], 0) + 1
        lines.append("Grades: " + ", ".join(f"{label} {count}" for label, count in counts.items()))
    reached = [e for e in events if e.get("reaches_floor") is not None]
    if reached:
        lines.append(f"Chest at or near floor (estimated): {sum(bool(e['reaches_floor']) for e in reached)} of {len(reached)}")
    lines += [
        "",
        "Limits: the chest position is estimated from pose, so it does not prove chest",
        "touched the floor. Grades compare each rep with the athlete's own calibration.",
        "Force, muscle tension and effort are not measured from video.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip", type=Path, help="Video file to analyze")
    parser.add_argument("--calibration", type=Path, help="Calibration JSON from calibrate.py")
    parser.add_argument("--layout", choices=LAYOUTS, default="reel")
    parser.add_argument("--static-camera", action="store_true",
                        help="Confirm the camera did not move during the set")
    args = parser.parse_args()

    clip = args.clip.resolve()
    if not clip.is_file():
        raise SystemExit(f"Clip not found: {clip}")
    calibration = json.loads(args.calibration.read_text(encoding="utf-8")) if args.calibration else None

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = OUTPUT_DIR / f"{stamp}-{clip.stem}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting pose (cached after the first run)...")
    track = load_pose_track(cached_track(clip, CACHE_DIR))
    tuning = PushupTuning.from_mapping({**DEFAULT_TUNING, "camera_stable_confirmed": args.static_camera})
    result = analyze_pushup_profile(track, tuning)
    events = result["events"]
    if calibration:
        apply_calibration(events, calibration)
    for event in events:
        event["reaches_floor"] = clears_floor(event.get("height_above_floor"))

    (run_dir / "reps.json").write_text(json.dumps(events, indent=2, ensure_ascii=False), encoding="utf-8")
    report = report_text(clip, events, calibration, tuning)
    (run_dir / "report.txt").write_text(report, encoding="utf-8")
    video = run_dir / f"analysis-{args.layout}.mp4"
    print(f"Rendering {args.layout} video...")
    render(clip, track, events, height_curve(track, tuning), calibration, args.layout, video)
    (run_dir / "run.json").write_text(json.dumps({
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "clip": str(clip),
        "clip_sha256": file_sha256(clip),
        "layout": args.layout,
        "calibration": calibration.get("name") if calibration else None,
        "parameters": tuning.as_dict(),
        "pose": track.metadata,
        "reps": len(events),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(report)
    print(f"Output: {run_dir}")


if __name__ == "__main__":
    main()