"""Self-contained Method 2 runner used by the official delivery entry point."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np

from bermguard_method2.tracking_adapter import TrackerAdapter
from method2_berm_height.height import BandMeasurement, height_interval, height_metres, measure_band, verdict
from method2_berm_height.scene import detect_boundaries, segment_for
from method2_final_osd.overlay import draw_overlay
from method2_ground_distance.geometry import CameraModel, Reference, Sample, clearance, fit_camera, ground_pose

HERE = Path(__file__).resolve().parent
DEFAULT_WEIGHT = HERE.parent / "models" / "yoloseg_maquinaria_train_v2.pt"
REFERENCES_FILE = HERE / "vehicle_references.json"


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BermGuard Method 2")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--berm-mask-root", type=Path)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--detect-every", type=int, default=1)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--fov-deg", type=float, default=70.0)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _references() -> dict[str, Reference]:
    raw = json.loads(REFERENCES_FILE.read_text(encoding="utf-8"))
    return {
        name: Reference(float(v["height_m"]), float(v["length_m"]), float(v["width_m"]),
                        float(v["uncertainty_fraction"]))
        for name, v in raw.items()
    }


def _class_name(value: str) -> str:
    compact = value.strip().lower().replace(" ", "_").replace("-", "_")
    if "bulld" in compact or "buld" in compact:
        return "Bulldozer"
    if "camion" in compact or "truck" in compact or "caex" in compact:
        return "Camion_Mina"
    return value.strip().replace(" ", "_") or "vehicle"


def _detections(result: Any, shape: tuple[int, int]) -> list[dict[str, Any]]:
    if result.boxes is None:
        return []
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    confidence = result.boxes.conf.detach().cpu().numpy()
    masks = None
    if result.masks is not None:
        masks = result.masks.data.detach().cpu().numpy()
    output = []
    for index, bbox in enumerate(boxes):
        mask = None
        if masks is not None and index < len(masks):
            mask = cv2.resize(masks[index], (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0.5
        output.append({
            "bbox": tuple(float(x) for x in bbox), "mask": mask,
            "class_name": _class_name(str(result.names[int(classes[index])])),
            "confidence": float(confidence[index]), "source": "yolo_seg_live",
        })
    return output


def _track_row(frame_index: int, timestamp: float, track: Any) -> dict[str, Any]:
    x1, y1, x2, y2 = track.bbox
    return {
        "frame_index": frame_index, "timestamp_s": timestamp, "track_id": int(track.track_id),
        "track_state": track.state.value, "position_source": track.position_source.value,
        "class_name": track.class_name, "confidence": float(track.confidence),
        "age_since_observation_s": float(track.age_since_observation_s),
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
    }


def _mask_path(root: Path | None, stem: str, frame_index: int) -> Path | None:
    if root is None:
        return None
    folder = root / stem
    candidates = (
        folder / f"human_mask_f{frame_index:05d}.png", folder / f"mask_{frame_index:05d}.png",
        folder / f"human_mask_f{frame_index:06d}.png", folder / f"mask_{frame_index:06d}.png",
    )
    return next((path for path in candidates if path.is_file()), None)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _charts(output: Path, tracks: list[list[dict[str, Any]]], distances: list[list[dict[str, Any]]],
            height_rows: list[dict[str, Any]]) -> None:
    points: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for frame_rows in tracks:
        for row in frame_rows:
            if row["position_source"] == "observed":
                points[int(row["track_id"])].append(((row["x1"] + row["x2"]) / 2, row["y2"]))
    fig, ax = plt.subplots(figsize=(10, 6))
    for track_id, values in sorted(points.items()):
        xy = np.asarray(values)
        ax.plot(xy[:, 0], xy[:, 1], ".-", markersize=2, label=f"ID {track_id}")
    ax.invert_yaxis(); ax.set_xlabel("x imagen (px)"); ax.set_ylabel("punto de contacto y (px)")
    ax.set_title("Distribución espacial observada de vehículos")
    if points: ax.legend(fontsize=6, ncol=3)
    fig.tight_layout(); fig.savefig(output / "vehicle_spatial_distribution.png", dpi=130); plt.close(fig)

    pair_min: dict[tuple[int, int], float] = {}
    ids = sorted(points)
    for frame_rows in distances:
        for row in frame_rows:
            if row.get("distance_m_low") is not None:
                key = tuple(sorted((int(row["track_a"]), int(row["track_b"]))))
                pair_min[key] = min(pair_min.get(key, float("inf")), float(row["distance_m_low"]))
                ids = sorted(set(ids) | set(key))
    matrix = np.full((len(ids), len(ids)), np.nan)
    lookup = {value: index for index, value in enumerate(ids)}
    for (a, b), value in pair_min.items():
        matrix[lookup[a], lookup[b]] = matrix[lookup[b], lookup[a]] = value
    fig, ax = plt.subplots(figsize=(7, 6))
    if ids:
        shown = np.ma.masked_invalid(matrix)
        image = ax.imshow(shown, cmap="RdYlGn", vmin=0, vmax=25)
        fig.colorbar(image, ax=ax, label="m")
    else:
        ax.text(0.5, 0.5, "Sin pares con geometría válida", ha="center", va="center")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks(range(len(ids)), ids); ax.set_yticks(range(len(ids)), ids)
    ax.set_title("Distancia mínima conservadora entre vehículos (m)")
    fig.tight_layout(); fig.savefig(output / "minimum_distance_matrix.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    valid = [r for r in height_rows if r.get("height_m") is not None]
    if valid: ax.scatter([r["timestamp_s"] for r in valid], [r["height_m"] for r in valid], s=10)
    ax.set_xlabel("Tiempo (s)"); ax.set_ylabel("Altura (m)"); ax.set_title("Altura visible del pretil")
    fig.tight_layout(); fig.savefig(output / "berm_height_over_time.png", dpi=130); plt.close(fig)


def _process(video: Path, output_root: Path, model: Any, options: argparse.Namespace,
             refs: dict[str, Reference]) -> dict[str, Any]:
    destination = output_root / video.stem / "method_2"
    destination.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"no se pudo abrir {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    tracker = TrackerAdapter(fps=fps)
    all_tracks: list[list[dict[str, Any]]] = []
    small_frames: list[np.ndarray] = []
    inference_seconds = 0.0; started = time.perf_counter(); index = 0
    while True:
        ok, frame = capture.read()
        if not ok: break
        small_frames.append(cv2.resize(frame, (320, 180)))
        detections: list[dict[str, Any]] = []
        if index % options.detect_every == 0:
            inference_started = time.perf_counter()
            result = model.predict(frame, device=options.device, conf=options.confidence, verbose=False)[0]
            inference_seconds += time.perf_counter() - inference_started
            detections = _detections(result, (height, width))
        tracks = tracker.update(detections, dt=1.0 / fps)
        all_tracks.append([_track_row(index, index / fps, track) for track in tracks])
        index += 1
    capture.release()
    actual_frames = index
    # Sólo corta un cambio estructural entre frames consecutivos.  El criterio acumulado por
    # defecto confunde un amanecer con un corte: sobre el video 02, que no tiene cortes, partía el
    # clip en once escenas, y cada trozo ajustaba su modelo de cámara con tan pocas muestras que
    # salían horizontes negativos y cámaras de veinte metros de altura.  El corte real del video
    # 01 se detecta igual, porque su señal entre frames consecutivos llega a 0,573.
    boundaries, scene_diagnostics = detect_boundaries(small_frames, reference_threshold=float("inf"))

    samples_by_scene: dict[int, list[Sample]] = defaultdict(list)
    heights_by_track: dict[int, list[float]] = defaultdict(list)
    for frame_index, rows in enumerate(all_tracks):
        scene = segment_for(frame_index, boundaries)
        for row in rows:
            if row["position_source"] != "observed" or row["class_name"] not in refs: continue
            box_height = float(row["y2"] - row["y1"]); heights_by_track[int(row["track_id"])].append(box_height)
            complete = row["x1"] > 1 and row["y1"] > 1 and row["x2"] < width - 2 and row["y2"] < height - 2
            samples_by_scene[scene].append(Sample(int(row["track_id"]), row["class_name"],
                                                   (row["x1"] + row["x2"]) / 2, row["y2"], box_height, complete))
    scene_models: dict[int, CameraModel] = {}
    scene_count = len(boundaries)
    for scene in range(scene_count):
        scene_models[scene] = fit_camera(samples_by_scene.get(scene, []), refs, height)

    distances_by_frame: list[list[dict[str, Any]]] = []
    height_rows: list[dict[str, Any]] = []
    minimum_distance = None; alert_frames = 0
    capture = cv2.VideoCapture(str(video))
    writer = cv2.VideoWriter(str(destination / "processed_with_osd.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame_index in range(actual_frames):
        ok, frame = capture.read()
        if not ok: break
        rows = all_tracks[frame_index]; scene = segment_for(frame_index, boundaries); camera = scene_models[scene]
        poses = []
        for row in rows:
            if row["position_source"] != "observed" or row["class_name"] not in refs: continue
            sample = Sample(int(row["track_id"]), row["class_name"], (row["x1"] + row["x2"]) / 2,
                            row["y2"], row["y2"] - row["y1"], True)
            median_height = float(np.median(heights_by_track[int(row["track_id"])]))
            pose = ground_pose(sample, refs[row["class_name"]], camera, width, options.fov_deg,
                               median_height_px=median_height)
            if pose.valid: poses.append(pose)
        distance_rows = []
        for a_index, a in enumerate(poses):
            for b in poses[a_index + 1:]:
                distance = clearance(a, b, refs)
                row = {"frame_index": frame_index, "timestamp_s": frame_index / fps,
                       "track_a": distance.a, "track_b": distance.b,
                       "distance_m_estimated": distance.nominal, "distance_m_low": distance.low,
                       "distance_m_high": distance.high, "state": distance.state,
                       "uncertain": distance.uncertain, "calibration_status": "estimated",
                       "observation_status": "observed", "reason": "ok"}
                distance_rows.append(row)
                if distance.low is not None:
                    minimum_distance = distance.low if minimum_distance is None else min(minimum_distance, distance.low)
        if any(r["distance_m_low"] is not None and r["distance_m_low"] <= 20 for r in distance_rows): alert_frames += 1
        distances_by_frame.append(distance_rows)

        mask_file = _mask_path(options.berm_mask_root, video.stem, frame_index)
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE) if mask_file else None
        if mask is None:
            height_row = {"frame_index": frame_index, "timestamp_s": frame_index / fps, "state": "N/D",
                          "reason": "sin_semilla_de_pretil", "height_px": None, "height_m": None,
                          "height_m_low": None, "height_m_high": None, "observation_status": "unknown"}
        else:
            if mask.shape != (height, width): mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            dynamic = np.zeros((height, width), np.uint8)
            for row in rows:
                if row["position_source"] == "observed":
                    cv2.rectangle(dynamic, (int(row["x1"]), int(row["y1"])), (int(row["x2"]), int(row["y2"])), 1, -1)
            measurement = measure_band(mask, dynamic)
            value, reason = (None, "sin_modelo_de_camara")
            low = high_value = None
            if measurement.thickness_px is not None and measurement.base_row is not None:
                value, reason = height_metres(measurement.thickness_px, measurement.base_row,
                                               camera.horizon_y, camera.camera_height_m)
                if value is not None:
                    low, high_value = height_interval(measurement.thickness_px, measurement.base_row,
                                                      camera.horizon_y - 10, camera.horizon_y + 10,
                                                      camera.camera_height_m, 0.15)
            height_row = {"frame_index": frame_index, "timestamp_s": frame_index / fps,
                          "state": "medida" if value is not None else "N/D", "reason": reason,
                          "height_px": measurement.thickness_px, "height_m": value,
                          "height_m_low": low, "height_m_high": high_value,
                          "observation_status": "observed"}
        height_rows.append(height_row)
        segment = None
        if height_row["height_m"] is not None:
            segment = {"estimate_m": height_row["height_m"], "low_m": height_row["height_m_low"],
                       "high_m": height_row["height_m_high"],
                       "verdict": verdict(height_row["height_m_low"], height_row["height_m_high"])}
        draw_overlay(frame, rows, distance_rows, height_row, segment, mask)
        writer.write(frame)
    capture.release(); writer.release()

    distance_flat = [row for group in distances_by_frame for row in group]
    distance_fields = ["frame_index", "timestamp_s", "track_a", "track_b", "distance_m_estimated",
                       "distance_m_low", "distance_m_high", "state", "uncertain", "calibration_status",
                       "observation_status", "reason"]
    _write_csv(destination / "distances.csv", distance_flat, distance_fields)
    _write_csv(destination / "berm_height_series.csv", height_rows,
               ["frame_index", "timestamp_s", "state", "reason", "height_px", "height_m",
                "height_m_low", "height_m_high", "observation_status"])
    _charts(destination, all_tracks, distances_by_frame, height_rows)
    elapsed = time.perf_counter() - started
    metadata = {
        "method": 2, "video": video.name, "frames": actual_frames, "reported_frame_count": frame_count,
        "resolution": [width, height], "source_fps": fps, "detect_every": options.detect_every,
        "device": options.device, "inference_seconds": inference_seconds,
        "processing_seconds": elapsed, "processing_ms_per_frame": elapsed * 1000 / max(actual_frames, 1),
        "average_processing_fps": actual_frames / max(elapsed, 1e-9),
        "minimum_conservative_distance_m": minimum_distance, "proximity_alert_frame_count": alert_frames,
        "proximity_alert": alert_frames > 0, "berm_mask_root": str(options.berm_mask_root) if options.berm_mask_root else None,
        "berm_status_without_seed": "N/D: sin_semilla_de_pretil",
        "camera_segments": [{"segment": i, "start_frame": boundaries[i], **model.__dict__}
                            for i, model in sorted(scene_models.items())],
        "scene_diagnostics": scene_diagnostics,
        "limitations": ["distancias estimadas con dimensiones de referencia y FOV supuesto",
                        "sin calibracion metrica externa", "el pretil requiere mascaras humanas externas"],
        "provenance": {
            "weight_sha256": _sha256(options.weights),
            "runner_sha256": _sha256(Path(__file__)),
            "vehicle_references_sha256": _sha256(REFERENCES_FILE),
        },
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[metodo 2] {video.name}: {actual_frames} frames, {metadata['average_processing_fps']:.2f} FPS")
    return metadata


def run(argv: Sequence[str] | None = None) -> int:
    options = _arguments(argv)
    if options.detect_every < 1:
        raise SystemExit("--detect-every debe ser >= 1")
    if not options.input.is_dir():
        raise SystemExit(f"no existe la entrada: {options.input}")
    if not options.weights.is_file():
        raise SystemExit(f"no existe el peso YOLO-seg: {options.weights}")
    videos = sorted(options.input.glob("*.mp4"), key=lambda p: p.name.lower())
    if not videos:
        raise SystemExit(f"no hay archivos .mp4 en {options.input}")
    options.output.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO
    model = YOLO(str(options.weights))
    refs = _references()
    summaries = [_process(video, options.output, model, options, refs) for video in videos]
    (options.output / "procedencia_metodo_2.json").write_text(
        json.dumps({"method": 2, "videos": summaries}, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
