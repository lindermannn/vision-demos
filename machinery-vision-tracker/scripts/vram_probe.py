#!/usr/bin/env python3
"""GPU memory probe for the benchmark, independent of any pipeline.

The benchmark module of the assessment asks for the trade-off between precision
and FPS/VRAM.  Method 1 runs through the ``bermguard`` pipeline; method 2 has its
own scripts.  This probe measures both without editing either, by wrapping the
*library* entry points that actually touch the GPU:

* ``grounding_dino``: ``transformers.GroundingDinoForObjectDetection.forward``
* ``sam2``: ``SAM2ImagePredictor`` and ``SAM2VideoPredictor`` inference methods
* ``yolo``: ``ultralytics.engine.model.Model.predict`` and ``track``

Two ways to use it:

    # Launch any script unmodified under the probe.
    python vram_probe.py --report vram.json --label metodo_1 --adapters grounding_dino \\
        -- main.py --input <videos> --output <salida> --method 1

    # Or from code.
    import vram_probe
    probe = vram_probe.get_probe()
    probe.install(["sam2", "yolo"])
    ...
    probe.write_report("vram.json", label="metodo_2")

What each number means, and what it does not:

* ``weights_mb``: bytes of CUDA parameters and buffers of the distinct models a
  component ran, counted once per storage even when two predictors share weights.
* ``activation_peak_mb``: the largest increase of PyTorch-allocated memory during a
  single call, above what was already allocated when the call started.  Calls
  nested inside another measured call are attributed to the outermost one.
* ``process.torch_peak_allocated_mb`` and ``torch_peak_reserved_mb``: process-wide
  peaks seen by the PyTorch caching allocator.  They exclude the CUDA context and
  some library workspaces, typically a few hundred MB.
* ``device``: an external NVML sample of the whole GPU.  Windows under the WDDM
  driver does not expose memory per process, so this is device-wide and includes
  every other application using the card; it is a cross-check, not a
  measurement of the method.

Without CUDA every VRAM field is ``null`` with an explicit basis; call counts and
times are still recorded.  ``null`` means "not applicable", never zero.
"""

from __future__ import annotations

import argparse
import functools
import importlib
import inspect
import itertools
import json
import os
import platform
import runpy
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

MB = 1024.0 * 1024.0
MARKER = "__vram_probe_wrapped__"

ADAPTERS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "grounding_dino": (
        ("transformers", "GroundingDinoForObjectDetection", "forward"),
    ),
    "sam2": (
        ("sam2.sam2_image_predictor", "SAM2ImagePredictor", "set_image"),
        ("sam2.sam2_image_predictor", "SAM2ImagePredictor", "set_image_batch"),
        ("sam2.sam2_image_predictor", "SAM2ImagePredictor", "predict"),
        ("sam2.sam2_image_predictor", "SAM2ImagePredictor", "predict_batch"),
        ("sam2.sam2_video_predictor", "SAM2VideoPredictor", "init_state"),
        ("sam2.sam2_video_predictor", "SAM2VideoPredictor", "add_new_points_or_box"),
        ("sam2.sam2_video_predictor", "SAM2VideoPredictor", "add_new_mask"),
        ("sam2.sam2_video_predictor", "SAM2VideoPredictor", "propagate_in_video"),
    ),
    "yolo": (
        ("ultralytics.engine.model", "Model", "predict"),
        ("ultralytics.engine.model", "Model", "track"),
    ),
}

LIBRARY_OF = {"grounding_dino": "transformers", "sam2": "sam2", "yolo": "ultralytics"}


@dataclass
class ComponentStats:
    calls: int = 0
    seconds: float = 0.0
    weights_bytes: int = 0
    activation_peak_bytes: int = 0
    total_peak_bytes: int = 0
    models: int = 0

    def as_report(self, cuda: bool) -> dict[str, Any]:
        def mb(value: int) -> Optional[float]:
            return round(value / MB, 1) if cuda else None

        return {
            "calls": self.calls,
            "seconds": round(self.seconds, 3),
            "ms_per_call": round(1000.0 * self.seconds / self.calls, 2) if self.calls else None,
            "models": self.models,
            "weights_mb": mb(self.weights_bytes),
            "activation_peak_mb": mb(self.activation_peak_bytes),
            "total_peak_mb": mb(self.total_peak_bytes),
        }


class DeviceSampler:
    """Whole-GPU memory sampled through NVML in a background thread."""

    def __init__(self, interval_s: float = 0.2) -> None:
        self.interval_s = interval_s
        self.basis = "unavailable"
        self.baseline_bytes: Optional[int] = None
        self.peak_bytes: Optional[int] = None
        self.total_bytes: Optional[int] = None
        self.name: Optional[str] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._nvml = None
        self._handle = None

    def start(self) -> None:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            name = pynvml.nvmlDeviceGetName(self._handle)
            self.name = name.decode() if isinstance(name, bytes) else str(name)
            self.total_bytes = int(info.total)
            self.baseline_bytes = self.peak_bytes = int(info.used)
            self.basis = "device_wide_includes_other_applications"
        except Exception:
            self._nvml = None
            return
        self._thread = threading.Thread(target=self._run, name="vram-probe-nvml", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def _sample(self) -> None:
        if self._nvml is None:
            return
        try:
            used = int(self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used)
        except Exception:
            return
        if self.peak_bytes is None or used > self.peak_bytes:
            self.peak_bytes = used

    def stop(self) -> None:
        self._sample()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass

    def as_report(self) -> dict[str, Any]:
        def mb(value: Optional[int]) -> Optional[float]:
            return None if value is None else round(value / MB, 1)

        delta = None
        if self.peak_bytes is not None and self.baseline_bytes is not None:
            delta = self.peak_bytes - self.baseline_bytes
        return {
            "basis": self.basis,
            "name": self.name,
            "total_mb": mb(self.total_bytes),
            "used_at_start_mb": mb(self.baseline_bytes),
            "used_peak_mb": mb(self.peak_bytes),
            "peak_above_start_mb": mb(delta),
        }


class Probe:
    def __init__(self, force_cpu: bool = False) -> None:
        self._lock = threading.RLock()
        self._local = threading.local()
        self.components: dict[str, ComponentStats] = {}
        self.installed: list[str] = []
        self.skipped: dict[str, str] = {}
        self._seen_storage: set[int] = set()
        self._seen_models: set[int] = set()
        self._peak_allocated = 0
        self._peak_reserved = 0
        self._torch = None
        self.cuda = False
        if not force_cpu:
            try:
                import torch  # type: ignore

                self._torch = torch
                self.cuda = bool(torch.cuda.is_available())
            except Exception:
                self._torch = None
        self.sampler = DeviceSampler()
        self.started_at = datetime.now(timezone.utc)

    # -- measurement ---------------------------------------------------------

    def _depth(self) -> int:
        return getattr(self._local, "depth", 0)

    def _begin(self) -> Optional[tuple[float, int]]:
        depth = self._depth()
        self._local.depth = depth + 1
        if depth > 0:
            return None
        start_allocated = 0
        if self.cuda:
            torch = self._torch
            torch.cuda.synchronize()
            # Keep whatever peak happened outside measured calls before resetting.
            self._peak_allocated = max(self._peak_allocated, torch.cuda.max_memory_allocated())
            self._peak_reserved = max(self._peak_reserved, torch.cuda.max_memory_reserved())
            start_allocated = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        return time.perf_counter(), start_allocated

    def _end(self, token: Optional[tuple[float, int]], component: str,
             owner: Any, record: bool) -> None:
        self._local.depth = max(0, self._depth() - 1)
        if token is None:
            return
        started, start_allocated = token
        elapsed = time.perf_counter() - started
        with self._lock:
            stats = self.components.setdefault(component, ComponentStats())
            peak = 0
            if self.cuda:
                torch = self._torch
                torch.cuda.synchronize()
                peak = int(torch.cuda.max_memory_allocated())
                self._peak_allocated = max(self._peak_allocated, peak)
                self._peak_reserved = max(self._peak_reserved, int(torch.cuda.max_memory_reserved()))
            if not record:
                return
            stats.calls += 1
            stats.seconds += elapsed
            if self.cuda:
                stats.activation_peak_bytes = max(stats.activation_peak_bytes, peak - start_allocated)
                stats.total_peak_bytes = max(stats.total_peak_bytes, peak)
                self._account_weights(stats, owner)

    def measure(self, component: str, owner: Any = None) -> "_Measurement":
        """Context manager for code that is not reached through an adapter."""

        return _Measurement(self, component, owner)

    def _account_weights(self, stats: ComponentStats, owner: Any) -> None:
        module = self._module_of(owner)
        if module is None or id(module) in self._seen_models:
            return
        size = 0
        on_cuda = False
        try:
            tensors = itertools.chain(module.parameters(), module.buffers())
            for tensor in tensors:
                if not tensor.is_cuda:
                    continue
                on_cuda = True
                try:
                    key = tensor.untyped_storage().data_ptr()
                except Exception:
                    key = tensor.data_ptr()
                if key in self._seen_storage:
                    continue
                self._seen_storage.add(key)
                size += tensor.numel() * tensor.element_size()
        except Exception:
            return
        # Weights still on the CPU are counted on a later call, once they moved.
        if on_cuda:
            self._seen_models.add(id(module))
            stats.models += 1
            stats.weights_bytes += size

    def _module_of(self, owner: Any) -> Any:
        if self._torch is None or owner is None:
            return None
        module_type = self._torch.nn.Module
        if isinstance(owner, module_type):
            return owner
        candidate = getattr(owner, "model", None)
        if isinstance(candidate, module_type):
            return candidate
        return None

    def _measured_generator(self, generator: Iterator[Any], component: str,
                            owner: Any) -> Iterator[Any]:
        while True:
            token = self._begin()
            try:
                item = next(generator)
            except StopIteration:
                self._end(token, component, owner, record=False)
                return
            except BaseException:
                self._end(token, component, owner, record=True)
                raise
            self._end(token, component, owner, record=True)
            yield item

    # -- wrapping ------------------------------------------------------------

    def wrap_method(self, owner_class: type, attribute: str, component: str) -> bool:
        original = owner_class.__dict__.get(attribute, getattr(owner_class, attribute))
        if getattr(original, MARKER, False):
            return False
        probe = self

        @functools.wraps(original)
        def wrapper(instance, *args, **kwargs):
            token = probe._begin()
            try:
                result = original(instance, *args, **kwargs)
            except BaseException:
                probe._end(token, component, instance, record=True)
                raise
            probe._end(token, component, instance, record=True)
            if inspect.isgenerator(result):
                return probe._measured_generator(result, component, instance)
            return result

        setattr(wrapper, MARKER, True)
        wrapper.__vram_probe_original__ = original  # type: ignore[attr-defined]
        setattr(owner_class, attribute, wrapper)
        return True

    def install(self, names: Iterable[str] = ("auto",)) -> list[str]:
        requested = list(names)
        if not requested or requested == ["auto"]:
            requested = [
                name for name, library in LIBRARY_OF.items()
                if importlib.util.find_spec(library) is not None
            ]
        for name in requested:
            if name not in ADAPTERS:
                self.skipped[name] = "unknown adapter"
                continue
            wrapped = 0
            for module_name, class_name, attribute in ADAPTERS[name]:
                try:
                    owner_class = getattr(importlib.import_module(module_name), class_name)
                    self.wrap_method(owner_class, attribute, name)
                    wrapped += 1
                except Exception as exc:  # the library may be absent or differ
                    self.skipped[f"{name}:{class_name}.{attribute}"] = f"{type(exc).__name__}: {exc}"
            if wrapped:
                self.installed.append(name)
        return list(self.installed)

    # -- report --------------------------------------------------------------

    def report(self, label: str = "", command: Sequence[str] = (), exit_code: Any = None) -> dict[str, Any]:
        torch = self._torch
        if self.cuda:
            torch.cuda.synchronize()
            self._peak_allocated = max(self._peak_allocated, int(torch.cuda.max_memory_allocated()))
            self._peak_reserved = max(self._peak_reserved, int(torch.cuda.max_memory_reserved()))
        cuda_device: dict[str, Any] = {"cuda_available": self.cuda}
        if torch is not None:
            cuda_device["torch"] = torch.__version__
            cuda_device["cuda"] = torch.version.cuda
        if self.cuda:
            props = torch.cuda.get_device_properties(0)
            cuda_device["name"] = props.name
            cuda_device["total_mb"] = round(props.total_memory / MB, 1)
        return {
            "schema": "bermguard_vram_probe_v1",
            "label": label,
            "command": list(command),
            "exit_code": exit_code,
            "started_at_utc": self.started_at.isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "host": {"platform": platform.platform(), "python": platform.python_version()},
            "cuda": cuda_device,
            "vram_basis": (
                "pytorch_caching_allocator" if self.cuda
                else "not_applicable_without_cuda"
            ),
            "adapters_installed": list(self.installed),
            "adapters_skipped": dict(self.skipped),
            "components": {
                name: stats.as_report(self.cuda) for name, stats in sorted(self.components.items())
            },
            "process": {
                "torch_peak_allocated_mb": round(self._peak_allocated / MB, 1) if self.cuda else None,
                "torch_peak_reserved_mb": round(self._peak_reserved / MB, 1) if self.cuda else None,
            },
            "device": self.sampler.as_report(),
            "definitions": {
                "weights_mb": "CUDA parameters and buffers of the models each component ran, "
                              "counted once per storage",
                "activation_peak_mb": "largest increase of allocated memory during one call "
                                      "above what was allocated when it started",
                "total_peak_mb": "largest process-wide allocated memory during one call",
                "process": "PyTorch caching allocator peaks; exclude the CUDA context",
                "device": "whole-GPU NVML sample including other applications; Windows "
                          "WDDM does not report memory per process",
                "null": "not applicable in this environment, never zero",
            },
        }

    def write_report(self, path: Path | str, **kwargs: Any) -> dict[str, Any]:
        report = self.report(**kwargs)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report


class _Measurement:
    def __init__(self, probe: Probe, component: str, owner: Any) -> None:
        self.probe, self.component, self.owner = probe, component, owner
        self.token: Optional[tuple[float, int]] = None

    def __enter__(self) -> "_Measurement":
        self.token = self.probe._begin()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.probe._end(self.token, self.component, self.owner, record=True)


_PROBE: Optional[Probe] = None


def get_probe() -> Probe:
    global _PROBE
    if _PROBE is None:
        _PROBE = Probe()
    return _PROBE


def run_target(target: Path, arguments: Sequence[str], report_path: Path, label: str,
               adapters: Sequence[str], probe: Optional[Probe] = None) -> int:
    """Run ``target`` as ``python target args`` under the probe; always write the report."""

    probe = probe or get_probe()
    target = target.resolve()
    probe.install(adapters)
    probe.sampler.start()
    saved_argv, saved_path = list(sys.argv), list(sys.path)
    sys.argv = [str(target), *arguments]
    sys.path.insert(0, str(target.parent))
    exit_code: Any = 0
    try:
        runpy.run_path(str(target), run_name="__main__")
    except SystemExit as exc:
        exit_code = exc.code if exc.code is not None else 0
    except BaseException:
        exit_code = 1
        raise
    finally:
        sys.argv, sys.path[:] = saved_argv, saved_path
        probe.sampler.stop()
        probe.write_report(report_path, label=label,
                           command=[str(target), *arguments], exit_code=exit_code)
    return exit_code if isinstance(exit_code, int) else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        print("uso: vram_probe.py --report <json> [--label L] [--adapters auto|a,b] -- <script> [args]",
              file=sys.stderr)
        return 2
    split = argv.index("--")
    parser = argparse.ArgumentParser(prog="vram_probe")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--adapters", default="auto")
    options = parser.parse_args(argv[:split])
    rest = argv[split + 1:]
    if not rest:
        print("vram_probe: falta el script a ejecutar despues de --", file=sys.stderr)
        return 2
    adapters = [item.strip() for item in options.adapters.split(",") if item.strip()]
    return run_target(Path(rest[0]), rest[1:], options.report, options.label, adapters)


if __name__ == "__main__":
    raise SystemExit(main())
