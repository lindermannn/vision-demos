# Limitations that must remain visible

- Method 2 height and distance outputs are estimates without defensible external calibration.
- SAM-2 propagates a prompt; it does not replace a trained berm detector.
- Accepted propagated masks are not automatically independent human ground truth.
- Validation using frames from the same video does not measure generalization to another camera.
- Synthetic videos support contract testing, not claims of real-site accuracy.
- Day/night transitions must not be confused with camera movement.
- Missing evidence should produce N/D, not a false alert or an unsupported green status.
