# Push-ups: how it works

This document explains the detection pipeline, its assumptions, and the limits you should
keep in mind when interpreting the output.

## Pipeline

1. **Pose extraction** (MediaPipe Pose, `model_complexity=1`, 20 fps)
   Frames are decoded with FFmpeg so rotation metadata is applied identically to the
   analysis renderer. Landmarks are normalized to the displayed frame size.

2. **Signal extraction** (`calisthenics.extract_signals`)
   - Elbow angle per side, averaged when both sides are visible.
   - `effort_score = flexion` (since push-ups don't use head/shoulder motion).
   - Visibility and landmark quality filters.

3. **Cycle detection** (`CycleStateMachine`)
   Deterministic state machine: rest → effort → peak → return.
   A repetition is `complete` only when:
   - ROM ≥ `minimum_rom_degrees` (default 30°)
   - Minimum elbow angle ≤ `flexed_angle_degrees` (default 105°)
   - Duration within `[minimum_duration_s, maximum_duration_s]` (default 0.35–8 s)
   - Visible fraction ≥ `minimum_visible_fraction` (default 0.70)

4. **Technical validation** (`BiomechanicsValidator`)
   - Only promotes to `valid` when the scene review confirms `floor_depth` and the
     elbow angle at peak is ≤ `flexed_angle`.
   - Without a scene review, every complete cycle stays `unverifiable`.

5. **Depth measurement** (`pushup_profiles._rep_depths`)
   Chest depth estimated from shoulder–elbow midpoint relative to the shoulder line,
   expressed in torso lengths. This is a *measurement for calibration*, not a
   chest-to-floor verdict.

6. **Calibration** (`pushup_calibration`)
   Athlete marks the deepest reps (`alto`) and the shallowest valid reps (`minimo`).
   Thresholds are derived from those human-verified reps; grades are relative to
   this athlete on this camera setup.

7. **Rendering** (`render.py`)
   Reel (1080×1920) or horizontal (1920×1080) video with skeleton overlay,
   live counter, depth graph, and grade badges. `peak` marks the lowest point for
   depth grading; the counter advances at `end`, after the ascent back to plank.

## Assumptions & limits

- **Elbow flexion ≠ chest to floor.** The detector sees elbow angle and shoulder
  vertical motion. It cannot see the chest or the floor. A `valid` grade means the
  elbow reached the configured angle *and* a human reviewer confirmed the chest
  depth in the scene review. Without review, reps are `unverifiable`.
- **Camera angle matters.** Front, side, and overhead views work if the elbows
  and shoulders are visible. Extreme angles where one arm is hidden will miss reps.
- **Camera must be static for clip-wide reference lines.** If the camera moves,
  only per-rep shoulder anchoring works (requires shoulders visible during the rep).
- **MediaPipe Pose landmarks are noisy.** Occlusion, clothing, and lighting cause
  jitter. The state machine smooths and filters, but false positives/negatives occur.
- **Grades are relative to the athlete's calibration.** A "ROM full" on one person
  with one camera is not comparable to another person or camera.
- **No force, tension, or effort inference.** The output is purely kinematic.

## Scene review format (optional)

If you want technical validation, provide a JSON with human observations:

```json
{
  "samples": [
    {
      "timestamp_s": 12.3,
      "camera_view": "front",
      "rules": {"floor_depth": true},
      "references": [{"kind": "floor", "visible": true}]
    }
  ]
}
```

Pass it to `analyze_track(track, "push_up", scene_review=review)` or include it
in your own workflow.

## Output files

- `reps.json` — one entry per detected rep with timestamps, depth, grade, and
  technical status.
- `report.txt` — human-readable summary.
- `run.json` — clip hash, parameters, pose backend, and rep count.
- `analysis-{layout}.mp4` — rendered video.
