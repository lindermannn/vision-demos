"""Web UI for push-up analysis: upload video -> get analysis video + JSON."""

from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "data" / "input"
OUTPUT_DIR = BASE_DIR / "data" / "output"
CACHE_DIR = BASE_DIR / "data" / "cache"
CALIB_DIR = BASE_DIR / "data" / "calibrations"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CALIB_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Push-up Analyzer")
jinja_env = Environment(loader=FileSystemLoader(str(BASE_DIR / "web" / "templates")), cache_size=0)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "web" / "static")), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


def run_analysis(video_path: Path, layout: str, static_camera: bool, calibration: str | None) -> Path:
    """Run main.py and return the output directory."""
    # Use the venv Python explicitly
    venv_python = BASE_DIR / ".venv" / "Scripts" / "python.exe"
    cmd = [
        str(venv_python), "main.py", str(video_path),
        "--layout", layout,
    ]
    if static_camera:
        cmd.append("--static-camera")
    if calibration:
        cmd.extend(["--calibration", calibration])

    result = subprocess.run(
        cmd,
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Analysis failed: {result.stderr}")

    # Parse output to find the created directory
    for line in result.stdout.splitlines():
        if line.startswith("Output: "):
            return Path(line.replace("Output: ", "").strip())
    raise RuntimeError("Could not find output directory in stdout")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    calibrations = [f.name for f in CALIB_DIR.glob("*.json")]
    template = jinja_env.get_template("index.html")
    content = template.render(request=request, calibrations=calibrations)
    return HTMLResponse(content=content)


@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    layout: str = Form("horizontal"),
    static_camera: bool = Form(False),
    calibration: str | None = Form(None),
):
    if not video.filename:
        raise HTTPException(400, "No file uploaded")

    # Save uploaded video
    ext = Path(video.filename).suffix
    uid = uuid.uuid4().hex[:8]
    input_path = INPUT_DIR / f"{uid}{ext}"

    with input_path.open("wb") as f:
        shutil.copyfileobj(video.file, f)

    try:
        # Resolve calibration path if provided
        calib_path = None
        if calibration:
            calib_path = str(CALIB_DIR / calibration)

        output_dir = run_analysis(input_path, layout, static_camera, calib_path)

        # Find generated files
        video_file = list(output_dir.glob(f"analysis-{layout}.mp4"))
        reps_file = output_dir / "reps.json"
        report_file = output_dir / "report.txt"
        run_file = output_dir / "run.json"

        return JSONResponse({
            "success": True,
            "output_dir": str(output_dir),
            "video_url": f"/outputs/{output_dir.name}/{video_file[0].name}" if video_file else None,
            "reps": json.loads(reps_file.read_text()) if reps_file.exists() else [],
            "report": report_file.read_text() if report_file.exists() else "",
            "run": json.loads(run_file.read_text()) if run_file.exists() else {},
        })
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/calibrations")
async def list_calibrations():
    return [f.name for f in CALIB_DIR.glob("*.json")]


@app.post("/calibrate")
async def calibrate(reference: UploadFile = File(...)):
    """Build calibration from reference JSON."""
    uid = uuid.uuid4().hex[:8]
    ref_path = CALIB_DIR / f"ref_{uid}.json"
    with ref_path.open("wb") as f:
        shutil.copyfileobj(reference.file, f)

    venv_python = BASE_DIR / ".venv" / "Scripts" / "python.exe"
    cmd = [str(venv_python), "calibrate.py", str(ref_path)]
    result = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise HTTPException(500, result.stderr)

    # Find created calibration file
    for line in result.stdout.splitlines():
        if line.startswith("saved: "):
            saved_path = Path(line.replace("saved: ", "").strip())
            return {"success": True, "calibration": saved_path.name, "output": result.stdout}
    return {"success": True, "output": result.stdout}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)