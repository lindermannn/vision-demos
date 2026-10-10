# How it works

## Counting reps

Pose comes from MediaPipe Pose, run locally on every frame of the clip. The result is a pose track: one set of landmarks per frame, with a visibility score for each joint. The detector combines three signals:

- Elbow angle, measured from shoulder, elbow and wrist on each side that is visible.
- Shoulder height in the image, with head height added as a fallback when the arms are hidden.
- Head height, from the nose or the ears.

A rep is a cycle that starts at rest, rises to a peak and returns to rest. Cycles must last between 0.45 s and 14 s. Peaks closer than 0.45 s to a counted peak are merged. Head and shoulder cycles only count when the arm cycle is missing or when a long arm cycle hides several peaks.

## Grading full or partial

Each counted rep gets a height: how far the head rises above the hands at its peak, in torso lengths. Hands are measured on the rep itself when they are visible. If they are hidden, the clip-wide hand line is used, and only when the camera was confirmed as static.

A height of 0 means the head reached the hands' line. Negative values mean the head stayed below it.

Each rep also carries a `reaches_bar` flag: true when the head is at or above the hands' line. It is an estimate from pose. It does not show that the chin cleared the bar, and its accuracy depends on the camera angle.

## Why calibrate

Heights depend on the athlete, the grip and the camera position, so no fixed cut-off works for everyone. The calibration uses reps the athlete has verified:

- The highest reps set the full high threshold.
- The lowest valid reps set the full threshold, minus a tolerance.
- The semi high threshold sits 20 % of the way from the full threshold to the full high threshold.

In the test series of 20 reps, with 10 more reps from a second clip used to lower the minimum, the calibration produced these values:

| Grade | From (torso lengths) |
|---|---|
| Full high | +0.39 |
| Semi high | +0.17 |
| Full | +0.05 |

The grades only compare a rep with the athlete's own reps. A new athlete or a different camera position needs a new calibration.

## The panel

The reel shows, in order:

- The set timer, starting at the first rep and stopping at the last.
- The count and the grade of the current rep.
- A graph of head height over the last few reps, with the calibration thresholds drawn across it.
- One bar per rep, coloured by grade.
- The rise, descent and hang time of the last rep.

The horizontal layout adds a minimap of the full set and a table of the last four reps.

The closing summary shows the total duration, the grade counts, the best rep, the average height, and a comparison between the first five and the last five reps.

## What the thresholds describe

The counting thresholds in `config.py` describe a complete pull-up: full range of motion, elbow flexed to about 105 degrees at the top, and a head rise that reaches the hands' line. A repetition that does not meet them is not counted.

The defaults were tuned on one athlete's sets and have not been tested on other athletes or camera setups yet. If a repetition that looks valid is missed, check the range and the rise in the video first. If it is still missed, the thresholds need a new profile in `config.py`.

Calibration only grades reps that were already counted. It does not change what counts as a rep.

## Limits

- One athlete per clip.
- Calibrations apply to the athlete and camera position they came from.
- Head height and `reaches_bar` do not show whether the chin clears the bar. A rep counted as movement is not a certified technique rep.
- Grip type, hang time and rise speed are measured from pose, not verified by a person.
- No measurement of force, muscle tension, calories or effort.
- Kipping and leg swing are not measured.
- The arm-based path depends on the elbow and wrist being visible. When they are hidden, the head and shoulder path can merge several reps into one long cycle.
