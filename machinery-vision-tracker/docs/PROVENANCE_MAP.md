# Provenance map

| Local source | Publication use | Handling |
|---|---|---|
| `02_METODO_1_FINAL` | Approved, frozen Method 1 | Extract code and edited documentation only |
| `03_METODO_2_FINAL` | Integrated runner and components | Publish custom code without weights or private outputs |
| `00_INDICE_Y_APRENDIZAJES` | Timeline and lessons | Adapt local paths and review confidentiality |
| `05_EVIDENCIA` | Provenance and validation | Preserve summaries, not original data |
| `07_EXPERIMENTOS` | Development history | Keep outside the public repository |
| `08_HERRAMIENTAS` | Local Label Studio/SAM-2 | Document process without environments or databases |
| `01_ENTREGA` | Submitted artifact | Do not copy to the public repository |

Publication retains enough provenance to explain the work without copying company data or recruitment materials.

## Authorized Method 2 screenshots

The owner authorized screenshots on October 5, 2026 and confirmed that the videos were AI-generated. Frames were extracted from existing outputs without repeating inference or changing overlays.

| Image in `docs/images/` | Source relative to `03_METODO_2_FINAL` | Time |
|---|---|---|
| `method2-video02-seeded-01s.png` | `salida_video02_con_mascaras/video_02/method_2/processed_with_osd.mp4` | 1 s |
| `method2-video02-seeded-03s.png` | Same run using approved masks | 3 s |
| `method2-video02-seeded-05s.png` | Same run using approved masks | 5 s |
| `method2-video02-04s.png` | `salidas_4_videos/video_02/method_2/processed_with_osd.mp4` | 4 s |

Their respective `metadata.json` files record 12.5081 FPS with masks and 25.7145 FPS without masks. Metrics are not mixed between runs. These are processing rates, not accuracy metrics.

The narrative adapts delivery documentation, the benchmark report, and curated records. Final geometry follows the report; the earlier global-scale reference in the delivery README is not reproduced here.

## Plots and height explanation

The three plots `video02-berm_height_over_time.png`, `video02-minimum_distance_matrix.png`, and `video02-vehicle_spatial_distribution.png` are unchanged copies of the corresponding files in `salida_video02_con_mascaras/video_02/method_2/`.

Height examples come from rows 0 and 239 of that run's `berm_height_series.csv`. Horizon and camera height come from its `metadata.json`. The formula, band measurement, interval, and verdict were checked against `metodo2/method2_berm_height/height.py`; class references were checked against benchmark section 3.2. Sprint 18 height results are identified as a different run and are not attributed to these plots.

Full videos, weights, and datasets are excluded. Visual selection does not demonstrate generalization or operational safety. Original artifact labels and technical paths are preserved for traceability.
