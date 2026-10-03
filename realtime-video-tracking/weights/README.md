# weights/

Model weights are **not committed** (`*.pt` is git-ignored). Put files here:

| file | what | where to get it |
|---|---|---|
| `yolov8s_traffic_best.pt` | YOLOv8s fine-tuned on VisDrone (optional) | not published - produce it with `notebooks/01_yolo_finetune_traffic.ipynb` |

```bash
python -m traffic_watch --source clip.mp4 --weights weights/yolov8s_traffic_best.pt --imgsz 960
```

The stock `yolov8s.pt` is downloaded automatically by Ultralytics. The OpenPSG checkpoint
(`third_party/openpsg/work_dirs/checkpoints/epoch_60.pth`) is described in the top-level README.
