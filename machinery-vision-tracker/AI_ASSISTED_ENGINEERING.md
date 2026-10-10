# AI-assisted engineering and MCP

## Principle

AI accelerated exploration, implementation, review, and documentation. Decisions affecting safety, metrics, and reproducibility were checked against code, fixtures, hashes, or human inspection.

## Concrete uses

| Area | Assistance | Applied control |
|---|---|---|
| Architecture | YOLO-seg/SAM-2/tracker pipeline comparison | Explicit contracts and limitations |
| Geometry | Homography, scale, and perspective review | Synthetic cases and rejection of extrapolation |
| Tracking | Hungarian, Kalman, and lifecycle proposals | Crossings, occlusions, and missing detections |
| Annotation | SAM-2 mask propagation in Label Studio | Human review and per-mask provenance |
| Automation | Packaging and validation scripts | Actual Docker execution and hashes during the original delivery |
| Research | Dimensional references and licensing | Recorded sources and declared assumptions |
| MCP | Tool, browser, and work-session control | Scoped actions without publishing credentials |

## Safety rules

- An agent's response is not evidence by itself.
- Results are checked against files, commands, tests, or images.
- SAM-2 proposals are distinguished from human-drawn masks by provenance.
- Private data is not published to make a demo more convincing.
- Inconclusive results remain N/D (not available).

## Multi-agent work

When different agents were used, responsibilities were documented and results were integrated only after reviewing interfaces, dependencies, and tests. This distinguishes AI acceleration from human accountability.
