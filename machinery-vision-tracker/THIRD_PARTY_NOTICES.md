# Third-party components

Custom source code is published under AGPL-3.0. Dependencies retain their original licenses; this publication does not relicense them. Their weights and vendored source are not included.

| Component | Use | License / source |
|---|---|---|
| Ultralytics YOLO11 | Machinery segmentation | [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) |
| SAM 2.1 | Mask proposals and propagation during development | [Apache-2.0 and associated notices](https://github.com/facebookresearch/sam2) |
| Grounding DINO | Method 1 detector | [Apache-2.0](https://github.com/IDEA-Research/GroundingDINO/blob/main/LICENSE) |
| Depth Anything V2 Small | Retained experimental adapter and asset tests; not the final Method 2 | [Apache-2.0 for Small](https://github.com/DepthAnything/Depth-Anything-V2) |
| PyTorch / torchvision | Inference | [BSD-3-Clause and associated notices](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| Transformers / Hugging Face Hub | Model loading | [Apache-2.0](https://github.com/huggingface/transformers/blob/main/LICENSE) |
| NumPy / SciPy | Computation and Hungarian assignment | BSD-3-Clause |
| OpenCV | Video and image processing | Apache-2.0; also review wheel components |
| Matplotlib | Plotting | Matplotlib license and associated notices |
| PyYAML | Configuration | MIT |
| Pillow | Images | MIT-CMU |

The public version integrates the two existing runners. SAM2 propagation was performed in the preparation workflow: the Method 2 runner receives external masks and does not execute SAM2 automatically. Without a seed, the berm remains N/D.

Checkpoints and data must be obtained separately with appropriate permissions and licenses. README images are authorized synthetic examples, not a distributed training dataset.
