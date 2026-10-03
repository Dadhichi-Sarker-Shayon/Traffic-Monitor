"""Everything that gets burned into the video: masks, tracks, zones, events."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .events import EngineConfig, Event, EventEngine
from .psg import PSGResult

# ------------------------------------------------------------------ palette
CLASS_COLORS = {
    "car": (200, 170, 60),        # blue-ish (BGR)
    "truck": (230, 130, 40),
    "bus": (200, 80, 200),
    "motorcycle": (60, 190, 230),
    "bicycle": (60, 220, 220),
    "person": (180, 90, 240),
}
EVENT_COLORS = {
    "JAM": (0, 140, 255),
    "JAM_ORIGIN": (0, 90, 255),
    "JAM_FRONT": (0, 200, 255),
    "ACCIDENT": (0, 0, 255),
    "STOPPED": (0, 220, 255),
    "PED_CONFLICT": (255, 0, 200),
}
STUFF_TINTS = {  # subtle background tints from PSG stuff masks
    "road": (40, 90, 40),
    "pavement-merged": (60, 60, 90),
    "sky-other-merged": (90, 60, 30),
}


def _instance_color(idx: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(idx * 9973 + 17)
    c = rng.integers(70, 255, size=3)
    return (int(c[0]), int(c[1]), int(c[2]))


def _put_label(img, text, org, color, *, scale=0.5, thickness=1, bg=None):
    x, y = org
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    if bg is not None:
        cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 3), bg, -1)
    cv2.putText(img, text, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
    return tw, th


# ------------------------------------------------------------------- layers
def draw_zones(img, engine: EventEngine, cfg: EngineConfig) -> None:
    ov = img.copy()
    for key, z in engine.zones.items():
        x1, y1, x2, y2 = engine.zone_rect(key)
        if z.jammed:
            color, alpha = (0, 0, 220), 0.22
        elif z.vehicles >= 1 and z.median_speed < cfg.jam_front_speed_px_s:
            color, alpha = (0, 180, 230), 0.14
        elif z.vehicles >= 1:
            color, alpha = (0, 200, 60), 0.10
        else:
            color, alpha = (160, 160, 160), 0.05
        cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        if z.vehicles >= 1:
            lbl = f"{z.vehicles}v {z.median_speed:.0f}px/s"
            _put_label(img, lbl, (x1 + 4, y1 + 16), (255, 255, 255), scale=0.42, bg=(40, 40, 40))
    # alpha-blend the tinted copy back (only where it actually changed)
    # only the zone region can have changed; working on that slice (and
    # masking with copyto rather than fancy indexing) keeps 4K frames from
    # allocating hundreds of MB of index arrays per frame
    _, ry1, _, ry2 = engine.roi
    ov_r, img_r = ov[ry1:ry2], img[ry1:ry2]
    changed = cv2.absdiff(ov_r, img_r).any(axis=2)
    blended = cv2.addWeighted(img_r, 0.55, ov_r, 0.45, 0)
    np.copyto(img_r, blended, where=changed[..., None])


def draw_psg(img, psg: PSGResult, *, show_masks=True, show_relations=True) -> None:
    h, w = img.shape[:2]
    # stuff tints (road etc.)
    if show_masks:
        for label, m in psg.stuff.items():
            tint = STUFF_TINTS.get(label)
            if tint is None:
                continue
            mask = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            tint_arr = np.array(tint, dtype=np.float32)
            img[mask] = (img[mask].astype(np.float32) * 0.75 + tint_arr * 0.25).astype(np.uint8)
        # instance masks + labels
        for inst in psg.instances:
            if inst.area < 400:
                continue
            mask = cv2.resize(inst.mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            col = np.array(_instance_color(inst.idx), dtype=np.float32)
            img[mask] = (img[mask].astype(np.float32) * 0.55 + col * 0.45).astype(np.uint8)
            x1, y1, x2, y2 = map(int, inst.bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), _instance_color(inst.idx), 1)
            _put_label(img, inst.label, (x1 + 2, y1 - 4 if y1 > 16 else y1 + 14),
                       _instance_color(inst.idx), scale=0.45, bg=(30, 30, 30))
    # relation edges: subject <-> object with predicate label
    if show_relations:
        for r in psg.top_relations(8):
            s, o = psg.instances[r.s], psg.instances[r.o]
            sc = (s.bbox[0] + s.bbox[2]) / 2, (s.bbox[1] + s.bbox[3]) / 2
            oc = (o.bbox[0] + o.bbox[2]) / 2, (o.bbox[1] + o.bbox[3]) / 2
            p1, p2 = (int(sc[0]), int(sc[1])), (int(oc[0]), int(oc[1]))
            cv2.line(img, p1, p2, (255, 255, 255), 1, cv2.LINE_AA)
            mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
            txt = r.predicate
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (mx - tw // 2 - 3, my - th - 4), (mx + tw // 2 + 3, my + 3), (35, 35, 35), -1)
            cv2.putText(img, txt, (mx - tw // 2, my), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (240, 240, 240), 1, cv2.LINE_AA)


def draw_tracks(img, engine: EventEngine, t: float, dets=None, *, px_per_meter=None) -> None:
    from .events import VEHICLE_CLASSES

    for tr in engine.tracks.values():
        if tr.cls not in VEHICLE_CLASSES | {"person"}:
            continue
        # a track not seen this frame (occlusion, a missed detection, the
        # tracker briefly losing it) still lives in engine.tracks for up to
        # history_s so the event engine can reason about it - but drawing its
        # last-known box here would paint a ghost rectangle that stays put
        # while the real vehicle has already moved on.
        if t - tr.last_seen > 0.5:
            continue
        x1, y1, x2, y2 = map(int, tr.box)
        col = CLASS_COLORS.get(tr.cls, (60, 220, 60))
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        speed = f"{tr.speed:.0f}"
        if px_per_meter:
            kmh = tr.speed / px_per_meter * 3.6
            speed = f"{kmh:.0f}km/h"
        tag = f"#{tr.tid} {tr.cls} {speed}p/s"
        if tr.cls == "person":
            tag = f"#{tr.tid} person"
        _put_label(img, tag, (x1, y1 - 6 if y1 > 20 else y1 + 15),
                   (255, 255, 255), scale=0.45, bg=col)


def draw_events(img, engine: EventEngine, frame_index: int) -> None:
    h, w = img.shape[:2]
    # active event banner list
    y = 26
    _put_label(img, f"traffic_watch  frame {frame_index}", (8, y), (255, 255, 255),
               scale=0.55, bg=(25, 25, 25))
    y += 24
    for ev in engine.active_events:
        col = EVENT_COLORS.get(ev.type, (200, 200, 200))
        txt = f"[!] {ev.type}: {ev.detail}" if ev.detail else f"[!] {ev.type}"
        _put_label(img, txt, (8, y), (255, 255, 255), scale=0.5, bg=col)
        y += 22

    # zone markers
    pulse = (math.sin(frame_index * 0.25) + 1) / 2.0
    if engine.jam_origin_key is not None:
        x1, y1, x2, y2 = engine.zone_rect(engine.jam_origin_key)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 90, 255), 3)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        radius = int(18 + 8 * pulse)
        cv2.circle(img, (cx, cy), radius, (0, 90, 255), 2)
        cv2.circle(img, (cx, cy), 4, (0, 90, 255), -1)
        _put_label(img, "JAM ORIGIN", (x1, y1 - 8 if y1 > 24 else y2 + 20),
                   (0, 90, 255), scale=0.6, thickness=2, bg=(20, 20, 20))
    for key, z in engine.zones.items():
        if not z.jammed:
            continue
        x1, y1, x2, y2 = engine.zone_rect(key)
        if engine.jam_origin_key == key:
            continue
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 140, 255), 2)

    # accident / stopped markers around their bboxes
    for ev in engine.active_events:
        if ev.bbox is None:
            continue
        col = EVENT_COLORS.get(ev.type, (0, 0, 255))
        x1, y1, x2, y2 = map(int, ev.bbox)
        for i in range(0, max(x2 - x1, y2 - y1), 14):  # dashed
            cv2.line(img, (x1 + i, y1), (min(x1 + i + 7, x2), y1), col, 3)
            cv2.line(img, (x1 + i, y2), (min(x1 + i + 7, x2), y2), col, 3)
        _put_label(img, ev.type, (x1, y1 - 8 if y1 > 24 else y2 + 18), col, scale=0.55,
                   thickness=2, bg=(20, 20, 20))
