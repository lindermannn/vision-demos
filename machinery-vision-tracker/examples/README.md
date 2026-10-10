# Bring your own inputs

Full videos are not distributed. Use your own videos or authorized material and obtain checkpoints separately.

```bash
python main.py --method 2 --input examples/videos --output outputs/demo --weights models/yoloseg_maquinaria_train_v2.pt --device cpu
```

Create `examples/videos` locally and place your clips there. The checkpoint must match the project's machinery classes. Without external berm masks, Method 2 reports N/D; it does not execute SAM2 automatically.
