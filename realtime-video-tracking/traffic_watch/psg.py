"""OpenPSG (PSGTR) adapter.

Loads the official PSGTR-r50 checkpoint through mmdet 2.x and turns the raw
``inference_detector`` output into a structured scene graph:

    PSGResult.instances  -> [{label, bbox, area, mask}]
    PSGResult.relations  -> [{s, o, predicate, score}]
    PSGResult.stuff      -> {label: mask}   (road, sky, pavement, ...)

Inference runs in a background thread; the video loop only submits frames
every ``interval`` frames and reads the latest finished result, so the
monitoring stays realtime even though PSG itself is slow.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OPENPSG_ROOT = REPO_ROOT / "third_party" / "openpsg"
DEFAULT_CFG = OPENPSG_ROOT / "configs" / "psgtr" / "psgtr_r50_psg_inference.py"
DEFAULT_CKPT = OPENPSG_ROOT / "work_dirs" / "checkpoints" / "epoch_60.pth"

# PSG panoptic classes (133) and predicates (56), as published in the
# official demo (huggingface.co/spaces/ECCV2022/PSG).
CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush', 'banner', 'blanket', 'bridge', 'cardboard',
    'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel',
    'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield',
    'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs',
    'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood',
    'water-other', 'window-blind', 'window-other', 'tree-merged', 'fence-merged',
    'ceiling-merged', 'sky-other-merged', 'cabinet-merged', 'table-merged',
    'floor-other-merged', 'pavement-merged', 'mountain-merged', 'grass-merged',
    'dirt-merged', 'paper-merged', 'food-other-merged', 'building-other-merged',
    'rock-merged', 'wall-other-merged', 'rug-merged', 'background',
]

PREDICATES = [
    'over', 'in front of', 'beside', 'on', 'in', 'attached to', 'hanging from',
    'on back of', 'falling off', 'going down', 'painted on', 'walking on',
    'running on', 'crossing', 'standing on', 'lying on', 'sitting on',
    'flying over', 'jumping over', 'jumping from', 'wearing', 'holding',
    'carrying', 'looking at', 'guiding', 'kissing', 'eating', 'drinking',
    'feeding', 'biting', 'catching', 'picking', 'playing with', 'chasing',
    'climbing', 'cleaning', 'playing', 'touching', 'pushing', 'pulling',
    'opening', 'cooking', 'talking to', 'throwing', 'slicing', 'driving',
    'riding', 'parked on', 'driving on', 'about to hit', 'kicking', 'swinging',
    'entering', 'exiting', 'enclosing', 'leaning on',
]

INSTANCE_OFFSET = 100000  # mmdet.datasets.coco_panoptic


@dataclass
class Instance:
    idx: int
    label: str
    cls_id: int
    bbox: Tuple[float, float, float, float]
    area: int
    mask: Optional[np.ndarray] = None  # bool (H, W), downsampled OK


@dataclass
class Relation:
    s: int   # instance idx (into PSGResult.instances)
    o: int
    predicate: str
    score: float


def road_mask_of(res: "PSGResult", height: int, width: int) -> Optional[np.ndarray]:
    """Boolean road mask for a scene-graph result, or None if it found no road.

    The event engine uses this to mean "cars standing on the road" literally:
    without it, cars parked in a side street or on a pavement inflate the
    vehicle count that the jam rule depends on.

    PSGTR reports the road as a panoptic *thing* as often as a stuff region, so
    both have to be consulted - looking only at `stuff` finds nothing on most
    real traffic footage.
    """
    roads = [m for label, m in res.stuff.items() if "road" in label.lower()]
    roads += [i.mask for i in res.instances
              if i.mask is not None and "road" in i.label.lower()]
    if not roads:
        return None
    mask = np.zeros(roads[0].shape[:2], dtype=bool)
    for m in roads:
        mask |= m.astype(bool)
    if height == mask.shape[0] and width == mask.shape[1]:
        return mask
    import cv2

    return cv2.resize(mask.astype(np.uint8), (width, height),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


@dataclass
class PSGResult:
    t: float = 0.0
    frame_index: int = -1
    infer_ms: float = 0.0
    instances: List[Instance] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    stuff: Dict[str, np.ndarray] = field(default_factory=dict)  # label -> bool mask
    caption: str = ""

    def top_relations(self, k: int = 8) -> List[Relation]:
        rels = sorted(self.relations, key=lambda r: -r.score)
        out = []
        for r in rels:
            if r.s < len(self.instances) and r.o < len(self.instances):
                out.append(r)
            if len(out) >= k:
                break
        return out

    def sentence(self, k: int = 6) -> str:
        parts = []
        for r in self.top_relations(k):
            s = self.instances[r.s].label
            o = self.instances[r.o].label
            parts.append(f"{s} {r.predicate} {o}")
        return "; ".join(parts)


class PSGRunner:
    """Owns the PSGTR model and runs inference off the main loop.

    Usage::

        runner = PSGRunner(interval=30)
        runner.start()
        ...
        if runner.should_submit(frame_index):
            runner.submit(frame_index, t, image)
        psg = runner.latest()          # Optional[PSGResult] or None
        runner.close()
    """

    def __init__(
        self,
        cfg_path: str | Path = DEFAULT_CFG,
        ckpt_path: str | Path = DEFAULT_CKPT,
        device: str = "cuda:0",
        interval: int = 30,          # run PSG every N frames (0 = disabled)
        num_rel: int = 12,           # relations kept per frame
        mask_size: int = 256,        # masks are stored downsampled to save RAM
        enabled: bool = True,
    ):
        self.cfg_path = Path(cfg_path)
        self.ckpt_path = Path(ckpt_path)
        self.device = device
        self.interval = interval
        self.num_rel = num_rel
        self.mask_size = mask_size
        self.enabled = enabled
        self.model = None
        self._lock = threading.Lock()
        self._pending = None          # (frame_index, t, image)
        self._latest: Optional[PSGResult] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._busy = False
        self.last_error: Optional[str] = None
        self.frames_done = 0

    # ------------------------------------------------------------- lifecycle
    def _ensure_model(self):
        if self.model is not None:
            return
        # mmdet 2.x lives in the mmcv-1.x venv; openpsg must be importable
        if str(OPENPSG_ROOT) not in sys.path:
            sys.path.insert(0, str(OPENPSG_ROOT))
        import openpsg.models  # noqa: F401  registers PSGTR & friends in the
        #                          mmdet registry (openpsg/__init__ is empty)
        from mmcv import Config
        from mmdet.apis import init_detector

        if not self.ckpt_path.exists():
            raise FileNotFoundError(
                f"PSGTR checkpoint missing: {self.ckpt_path}"
            )
        cfg = Config.fromfile(str(self.cfg_path))
        self.model = init_detector(cfg, str(self.ckpt_path), device=self.device)

    def start(self):
        if not self.enabled or self.interval <= 0:
            return
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._worker, name="psg", daemon=True)
            self._thread.start()

    def close(self):
        self._stop.set()
        with self._lock:
            self._pending = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    # ----------------------------------------------------------------- queue
    def should_submit(self, frame_index: int) -> bool:
        if not self.enabled or self.interval <= 0:
            return False
        if frame_index % self.interval != 0:
            return False
        with self._lock:
            return not self._busy and self._pending is None

    def submit(self, frame_index: int, t: float, image_bgr: np.ndarray):
        with self._lock:
            self._pending = (frame_index, t, image_bgr.copy())

    def latest(self) -> Optional[PSGResult]:
        with self._lock:
            return self._latest

    def _worker(self):
        while not self._stop.is_set():
            with self._lock:
                job = self._pending
                self._pending = None
                self._busy = job is not None
            if job is None:
                self._stop.wait(0.05)
                continue
            frame_index, t, image = job
            try:
                res = self.infer(image, t=t, frame_index=frame_index)
                with self._lock:
                    self._latest = res
                    self.frames_done += 1
            except Exception as e:  # keep the monitor alive no matter what
                self.last_error = f"{type(e).__name__}: {e}"
            finally:
                with self._lock:
                    self._busy = False

    # -------------------------------------------------------------- inference
    def infer(self, image_bgr: np.ndarray, *, t: float = 0.0, frame_index: int = -1) -> PSGResult:
        """Synchronous single-image inference (also used by the worker)."""
        from mmdet.apis import inference_detector

        self._ensure_model()
        t0 = time.perf_counter()
        raw = inference_detector(self.model, image_bgr)
        ms = (time.perf_counter() - t0) * 1000.0
        return parse_psgtr_result(raw, image_bgr.shape[:2], t=t, frame_index=frame_index, infer_ms=ms,
                                  num_rel=self.num_rel, mask_size=self.mask_size)


def parse_psgtr_result(
    raw,
    hw: Tuple[int, int],
    *,
    t: float,
    frame_index: int,
    infer_ms: float,
    num_rel: int = 12,
    mask_size: int = 256,
) -> PSGResult:
    """Convert the mmdet 2.x PSGTR output (SegDataChunk-like) into PSGResult.

    Field layout follows the official demo (utils.py): ``pan_results``
    (H,W) panoptic map; ``masks`` (N,H,W); ``labels`` (N,) 1-based class ids;
    ``rel_pair_idxes`` (R,2); ``rel_dists`` (R,57) with column 0 = background.
    """
    h, w = hw
    out = PSGResult(t=t, frame_index=frame_index, infer_ms=infer_ms)

    # ---- instances -------------------------------------------------------
    masks = getattr(raw, "masks", None)
    labels = getattr(raw, "labels", None)
    if masks is None or len(masks) == 0:
        return out
    masks = np.asarray(masks)
    if masks.dtype != np.bool_:
        masks = masks > 0.5
    labels = np.asarray(labels).astype(int).reshape(-1)
    if len(labels) != len(masks):
        labels = labels[: len(masks)]

    inst: List[Instance] = []
    for i, (m, lb) in enumerate(zip(masks, labels)):
        cls_id = int(lb) - 1  # PSGTR labels are 1-based (0 == background)
        if not (0 <= cls_id < len(CLASSES)) or CLASSES[cls_id] == "background":
            continue
        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            continue
        # OpenPSG returns an all-ones full-frame mask when *nothing* passed
        # its score thresholds — that is an empty result, not an instance.
        if len(ys) >= 0.95 * h * w and len(masks) == 1:
            continue
        bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
        small = _resize_mask(m, (mask_size, mask_size))
        inst.append(Instance(idx=len(inst), label=CLASSES[cls_id], cls_id=cls_id,
                             bbox=bbox, area=int(len(ys)), mask=small))
    out.instances = inst

    # old mask index -> compacted index (some masks may have been dropped)
    remap = {}
    kept = []
    for i, lb in enumerate(labels):
        cls_id = int(lb) - 1
        if 0 <= cls_id < len(CLASSES) and CLASSES[cls_id] != "background":
            kept.append(i)
    for new_i, old_i in enumerate(kept):
        remap[old_i] = new_i

    # ---- relations -------------------------------------------------------
    pairs = getattr(raw, "rel_pair_idxes", None)
    dists = getattr(raw, "rel_dists", None)
    if pairs is not None and dists is not None and len(pairs):
        pairs = np.asarray(pairs).astype(int)
        dists = np.asarray(dists)
        rel_scores = dists[:, 1:]  # drop background column
        best = rel_scores.max(axis=1)
        k = min(num_rel, len(best))
        if k > 0:
            top = np.argpartition(best, -k)[-k:]
            for r in top:
                s, o = int(pairs[r][0]), int(pairs[r][1])
                if s not in remap or o not in remap or remap[s] < 0 or remap[o] < 0:
                    continue
                if remap[s] == remap[o]:
                    continue
                pred_id = int(rel_scores[r].argmax())
                if not (0 <= pred_id < len(PREDICATES)):
                    continue
                out.relations.append(
                    Relation(s=remap[s], o=remap[o], predicate=PREDICATES[pred_id],
                             score=float(best[r]))
                )

    # ---- stuff (road/sky/...) from panoptic map --------------------------
    pan = getattr(raw, "pan_results", None)
    if pan is not None:
        pan = np.asarray(pan)
        for pid in np.unique(pan):
            pid = int(pid)
            if pid == 133:  # VOID
                continue
            cls_id = pid % INSTANCE_OFFSET
            if 0 <= cls_id < len(CLASSES):
                label = CLASSES[cls_id]
                if pid < INSTANCE_OFFSET:  # stuff classes have no instance id
                    m = pan == pid
                    if m.any():
                        out.stuff[label] = _resize_mask(m, (mask_size, mask_size))

    out.caption = out.sentence()
    return out


def _resize_mask(m: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    import cv2

    sw, sh = size
    return cv2.resize(m.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST).astype(bool)
