"""Build a height calibration from reps the athlete has verified.

Usage:
    python calibrate.py reference.json

reference.json example:
    {
      "name": "my-pushups",
      "tolerance": 0.02,
      "semi_fraction": 0.2,
      "references": [
        {"clip": "../data/input/set-of-20.mp4", "alto": "1-5", "minimo": "16-20",
         "all_valid": true, "static_camera": true}
      ]
    }

"alto" are the highest reps, "minimo" the lowest valid reps. Paths are relative
to the reference file. The result is saved under data/calibrations/<name>.json.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from config import CACHE_DIR, CALIBRATION_DIR, DEFAULT_TUNING
from pushup_demo.extract import cached_track
from pushup_demo.pose_contract import load_pose_track
from pushup_demo.pushup_calibration import build_calibration, parse_reps
from pushup_demo.pushup_profiles import PushupTuning, analyze_pushup_profile


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")[:50]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.reference.read_text(encoding="utf-8"))
    name = spec["name"].strip()
    if len(name) < 3 or not slug(name):
        raise SystemExit("name must have at least three letters")
    references = []
    for item in spec["references"]:
        clip = (args.reference.parent / item["clip"]).resolve()
        track = load_pose_track(cached_track(clip, CACHE_DIR))
        profile = PushupTuning.from_mapping({**DEFAULT_TUNING,
                                             "camera_stable_confirmed": bool(item.get("static_camera", False))})
        events = analyze_pushup_profile(track, profile)["events"]
        print(f"{clip.name}: {len(events)} reps detected")
        references.append({
            "events": events,
            "alto": parse_reps(item.get("alto", ""), len(events)),
            "minimo": parse_reps(item.get("minimo", ""), len(events)),
            "validas": range(1, len(events) + 1) if item.get("all_valid", True) else (),
        })
    calibration = build_calibration(references, tolerance=float(spec.get("tolerance", .02)),
                                    semi_fraction=float(spec.get("semi_fraction", .2)))
    calibration = {**calibration, "name": name, "slug": slug(name),
                   "references": [{key: item[key] for key in item if key != "clip"}
                                  for item in spec["references"]]}
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    target = CALIBRATION_DIR / f"{slug(name)}.json"
    target.write_text(json.dumps(calibration, indent=2, ensure_ascii=False), encoding="utf-8")
    print("thresholds:", json.dumps(calibration["thresholds"]))
    print(f"saved: {target}")


if __name__ == "__main__":
    main()