# Push-ups

Push-up rep counting, timing and depth grading from pose estimation. Runs locally with MediaPipe Pose. The live counter advances only when the athlete rises back to plank; the lowest point is used for depth grading, not for counting.

Run the commands below from this folder (`push_ups`).

| Project | What it does | Key model |
|---|---|---|
| Push-ups | Counts reps, times each one, grades how deep it goes against the athlete's own calibration, and renders a reel or horizontal analysis video | MediaPipe Pose (local) |

Details and limits are in [how_it_works.md](how_it_works.md).
Summary screens and HUD design details are in [docs/SUMMARY_REDESIGN.md](docs/SUMMARY_REDESIGN.md).

Sample videos and generated outputs are local data and are not included in the repository.

## Run it

1. Create the environment (Python 3.12):

       python -m venv .venv
       .venv\Scripts\activate
       pip install -r requirements.txt

2. Install FFmpeg and make sure `ffmpeg` and `ffprobe` are on your PATH.

3. Download Google's [Pose Landmarker Lite model](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task) and save it as `push_ups/pose_landmarker_lite.task`. The model asset is not bundled in this repository.

4. Put a clip in `data/input/`. Film from the front, the side, or overhead, with the whole body in frame. The first run extracts the pose and caches it in `data/cache/`.

5. Optional: build a calibration from reps you know are valid. Describe them in a JSON file and run:

       python calibrate.py reference.json

   The example file is in `docs/calibration-example.json`. Use the deepest reps for `alto`, the shallowest valid reps for `minimo`, and set `static_camera` only if the camera did not move during the set.

6. Analyze a clip:

       python main.py data/input/set.mp4
       python main.py data/input/set.mp4 --calibration data/calibrations/mis-flexiones-demo.json --layout horizontal
       python main.py data/input/set.mp4 --static-camera

   `--layout` is `reel` (1080×1920, default) or `horizontal` (1920×1080).

## Output

Each run creates a folder under `data/output/`:

    data/output/20260610-203000-set/
    ├── analysis-reel.mp4   # analysis video with skeleton, counter and grades
    ├── reps.json           # one entry per rep: start, peak, end, depth, grade
    ├── report.txt          # summary of the set
    └── run.json            # clip hash, parameters and pose backend used

`data/` is ignored by git, so clips, cached poses and calibrations stay on your machine.

## Tests

    python -m unittest tests.test_pushup_demo

## License

Apache-2.0. See [LICENSE](LICENSE).
