"""Smoke test: does YOLOv8 actually detect the synthetic traffic? Compares
its per-frame detections against the ground-truth JSONL."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from traffic_watch.detector import Detector  # noqa: E402


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter)


def main() -> int:
    video = ROOT / "sample_media" / "clips" / "demo_traffic.mp4"
    gt_path = ROOT / "sample_media" / "clips" / "demo_gt.jsonl"
    gt = {}
    with open(gt_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            gt[rec["frame"]] = rec["dets"]

    cap = cv2.VideoCapture(str(video))
    det = Detector(weights="yolov8s.pt", device="cuda:0")

    step = 30
    tot_gt = tot_hit = tot_pred = 0
    id_switch = 0
    prev_ids = set()
    for fi in range(0, 1200, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            break
        preds = det.track(frame)
        gts = [d for d in gt.get(fi, []) if d["cls"] != "person"]
        hits = 0
        for g in gts:
            if any(iou(g["bbox"], p.bbox) > 0.4 for p in preds):
                hits += 1
        tot_gt += len(gts)
        tot_hit += hits
        tot_pred += sum(1 for p in preds if p.cls != "person")
        ids = {p.id for p in preds}
        if prev_ids and ids:
            id_switch += len(prev_ids - ids)
        prev_ids = ids
        print(f"f{fi:4d}  gt={len(gts):2d}  pred={sum(1 for p in preds if p.cls!='person'):2d}  hits={hits:2d}"
              + (f"  person={sum(1 for p in preds if p.cls=='person')}" if any(d['cls']=='person' for d in gt.get(fi,[])) else ""))
    cap.release()
    recall = tot_hit / max(1, tot_gt)
    precision_like = tot_hit / max(1, tot_pred)
    print(f"\nvehicle recall={recall:.2%}  precision-ish={precision_like:.2%}  "
          f"id-churn per step={id_switch/40:.1f}")
    return 0 if recall > 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
