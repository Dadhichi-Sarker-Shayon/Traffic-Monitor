"""YOLOv8 + ByteTrack wrapper — persistent IDs for vehicles & pedestrians."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

# COCO ids we care about for traffic monitoring
COCO_VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 1: "bicycle"}
COCO_PERSONS = {0: "person"}
COCO_KEEP = set(COCO_VEHICLES) | set(COCO_PERSONS)


@dataclass
class Detection:
    id: int
    cls: str
    bbox: tuple  # xyxy
    conf: float


class Detector:
    """Wraps ``ultralytics`` YOLO with built-in ByteTrack.

    ``model.track(..., persist=True)`` keeps tracker state between frames,
    giving stable IDs across the whole video — which is what the event
    engine needs to reason about motion over time.
    """

    def __init__(
        self,
        weights: str = "yolov8s.pt",
        device: Optional[str] = None,
        conf: float = 0.4,
        iou: float = 0.6,
        max_det: int = 100,
        half: bool = True,
    ):
        from ultralytics import YOLO  # lazy: heavy import

        self.model = YOLO(weights)
        if device is not None:
            self.model.to(device)
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.half = half
        self._classes = sorted(COCO_KEEP)

    def track(self, frame: np.ndarray) -> List[Detection]:
        res = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            classes=self._classes,
            verbose=False,
        )[0]
        out: List[Detection] = []
        if res.boxes is None or len(res.boxes) == 0:
            return out
        boxes = res.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        ids = (
            boxes.id.cpu().numpy().astype(int)
            if boxes.id is not None
            else np.arange(len(xyxy))
        )
        for i in range(len(xyxy)):
            coco_id = int(clss[i])
            name = COCO_VEHICLES.get(coco_id) or COCO_PERSONS.get(coco_id)
            if name is None:
                continue
            out.append(
                Detection(
                    id=int(ids[i]),
                    cls=name,
                    bbox=tuple(map(float, xyxy[i])),
                    conf=float(confs[i]),
                )
            )
        return out

    def reset(self) -> None:
        self.model.predictor = None  # drops tracker state on source switch
