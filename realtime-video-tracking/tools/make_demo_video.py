"""Generate a synthetic traffic clip that exercises every event type.

Timeline (30 fps, 1280x720, 40 s):
    0-4s    free flow, three lanes
    ~4s     lane-0 leader brakes hard; chaser rear-ends it   -> ACCIDENT
    4-34s   traffic queues behind the crash                  -> JAM + JAM_ORIGIN
    22-30s  pedestrian crosses the roadway                   -> PED_CONFLICT
    34s     wreck is towed away, queue discharges            -> JAM clears

Also writes ground-truth detections (``demo_gt.jsonl``) so the pipeline can be
verified even without YOLO (``--detections path``).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 1280, 720, 30
DUR = 40.0
LANES = [440, 520, 600]        # y of lane centers (bottom half = ROI)
CAR_COLORS = [(200, 70, 60), (60, 160, 220), (230, 230, 230), (60, 200, 120),
              (40, 40, 40), (240, 170, 40), (150, 70, 200), (90, 190, 230)]
CRASH_T, CRASH_X = 4.0, 760
TOW_T = 34.0
PED = {"x": 870, "y0": 350, "cross_from": 22.0, "cross_to": 30.0}


def draw_car(img, x, y, w=96, h=44, color=(200, 70, 60), brake=False):
    """A simple but car-like vehicle: body, cabin, windows, wheels, lights."""
    x, y = int(x), int(y)
    if x > W + 120 or x < -120:
        return
    y1, y2 = max(0, y + h - 6), min(H, y + h + 6)
    x1, x2 = max(0, x), min(W, x + w)
    if y2 > y1 and x2 > x1:
        img[y1:y2, x1:x2] = (img[y1:y2, x1:x2] * 0.7).astype(np.uint8)
    cv2.rectangle(img, (x + 8, y + h - 8), (x + 26, y + h + 2), (25, 25, 25), -1)
    cv2.rectangle(img, (x + w - 26, y + h - 8), (x + w - 8, y + h + 2), (25, 25, 25), -1)
    cv2.rectangle(img, (x, y + 8), (x + w, y + h - 4), color, -1)
    cv2.rectangle(img, (x, y + 8), (x + w, y + h - 4), tuple(int(c * 0.6) for c in color), 2)
    cx1 = x + int(w * 0.28)
    cv2.rectangle(img, (cx1, y), (x + int(w * 0.78), y + 18), color, -1)
    cv2.rectangle(img, (cx1 + 2, y + 3), (cx1 + int(w * 0.2), y + 15), (180, 190, 200), -1)
    cv2.rectangle(img, (x + int(w * 0.55), y + 3), (x + int(w * 0.76), y + 15), (170, 180, 195), -1)
    cv2.rectangle(img, (x + w - 6, y + 14), (x + w, y + 20), (120, 240, 255), -1)
    tail = (0, 0, 255) if brake else (40, 40, 255)
    cv2.rectangle(img, (x, y + 14), (x + 6, y + 20), tail, -1)


def draw_person(img, x, y, w=24, h=56, color=(60, 60, 200)):
    x, y = int(x), int(y)
    cv2.circle(img, (x + w // 2, y + 8), 8, (90, 160, 220), -1)
    cv2.rectangle(img, (x + 5, y + 16), (x + w - 5, y + 40), color, -1)
    stride = 4 if (int(x) // 6) % 2 == 0 else -4
    cv2.line(img, (x + 9, y + 40), (x + 7 - stride // 2, y + h), (30, 30, 30), 4)
    cv2.line(img, (x + w - 9, y + 40), (x + w - 7 + stride // 2, y + h), (30, 30, 30), 4)


def road_background():
    img = np.full((H, W, 3), (70, 72, 74), np.uint8)
    img[:340] = (96, 120, 96)
    cv2.rectangle(img, (0, 330), (W, 346), (150, 150, 150), -1)
    cv2.rectangle(img, (0, 380), (W, H), (85, 87, 90), -1)
    for y in (480, 560):
        for x in range(0, W, 90):
            cv2.rectangle(img, (x, y - 3), (x + 45, y + 3), (230, 230, 230), -1)
    cv2.rectangle(img, (0, 396), (W, 404), (230, 230, 230), -1)
    cv2.rectangle(img, (0, H - 30), (W, H - 22), (230, 230, 230), -1)
    return img


class Car:
    def __init__(self, tid, lane, x, v, color):
        self.tid, self.lane, self.x, self.v0 = tid, lane, x, v
        self.v = float(v)
        self.color = color
        self.y = LANES[lane] - 22
        self.brake = False
        self.dead = False        # towed away / wrapped

    @property
    def box(self):
        return (self.x, self.y, self.x + 96, self.y + 44)


def build_scenario():
    cars = []
    # lane 0: slow lane; leader will brake at CRASH_T, chaser behind it
    cars.append(Car(1, 0, 120, 6, CAR_COLORS[0]))        # leader
    cars.append(Car(2, 0, -80, 6, CAR_COLORS[1]))        # chaser (gap 200)
    for i, x0 in enumerate([-330, -580, -830, -1080, -1330]):
        cars.append(Car(3 + i, 0, x0, 6, CAR_COLORS[(i + 2) % len(CAR_COLORS)]))
    # lanes 1 & 2: fast flowing traffic
    for lane in (1, 2):
        for i in range(8):
            x0 = -40 - 200 * i
            cars.append(Car(20 + lane * 10 + i, lane, x0, 12 + 0.5 * i,
                            CAR_COLORS[(lane + i) % len(CAR_COLORS)]))
    return cars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sample_media/clips/demo_traffic.mp4")
    ap.add_argument("--gt", default="sample_media/clips/demo_gt.jsonl")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

    cars = build_scenario()
    bg = road_background()
    gt_f = open(args.gt, "w", encoding="utf-8")
    next_tid = 100                     # for wrap re-entries (fresh track ids)
    n_frames = int(DUR * FPS)

    for fi in range(n_frames):
        t = fi / FPS
        img = bg.copy()

        lane0 = sorted([c for c in cars if c.lane == 0 and not c.dead],
                       key=lambda c: -c.x)  # front-most first
        lead = next((c for c in lane0 if c.tid == 1), None)
        chaser = next((c for c in lane0 if c.tid == 2), None)

        # ---- leader: brake into the crash --------------------------------
        if lead is not None:
            if t >= CRASH_T - 0.6:
                lead.brake = True
                lead.v = max(0.0, lead.v0 * (1 - (t - (CRASH_T - 0.6)) / 0.6))
            if t >= TOW_T:
                lead.v = 14.0  # towed away
            lead.x += lead.v

        # ---- chaser: keeps speed, then rear-ends --------------------------
        if chaser is not None:
            if t >= TOW_T:
                chaser.v = 14.0
                chaser.x += chaser.v
            elif lead is not None and t >= CRASH_T - 0.6 and chaser.x + 96 >= lead.x - 6:
                chaser.v = 0.0
                chaser.brake = True
                chaser.x = lead.x - 40          # resting overlap (IoU ~0.4)
            else:
                chaser.v = chaser.v0 if t < CRASH_T - 0.6 else chaser.v0 * 0.9
                chaser.brake = t >= CRASH_T - 0.6
                chaser.x += chaser.v

        # ---- followers: car-following, queue behind the crash -------------
        for c in lane0:
            if c.tid in (1, 2):
                continue
            ahead = min((o for o in lane0 if o.x > c.x), key=lambda o: o.x - c.x, default=None)
            gap = (ahead.x - c.x) if ahead is not None else 9999
            if t >= TOW_T:
                c.v = min(c.v0, c.v + 0.5)      # wreck cleared, accelerate
                c.brake = False
            elif gap < 130:
                c.v = max(0.0, min(c.v, 0.0 if gap < 118 else 2.5))
                c.brake = True
            elif gap < 260:
                c.v = min(c.v, c.v0 * 0.5)
                c.brake = True
            else:
                c.v = min(c.v0, c.v + 0.3)
                c.brake = False
            c.x += c.v

        # ---- lanes 1 & 2 flow + wrap (fresh id on re-entry) ---------------
        for c in cars:
            if c.lane == 0 or c.dead:
                continue
            c.x += c.v
            if c.x > W + 150:
                c.x = -260 - (next_tid % 5) * 40
                c.tid = next_tid
                next_tid += 1

        # ---- draw ---------------------------------------------------------
        for c in cars:
            if c.dead:
                continue
            draw_car(img, c.x, c.y, color=c.color, brake=c.brake)

        # ---- pedestrian crossing 22s-30s ---------------------------------
        ped_dets = []
        if PED["cross_from"] <= t <= PED["cross_to"]:
            frac = (t - PED["cross_from"]) / (PED["cross_to"] - PED["cross_from"])
            py = int(PED["y0"] + frac * (660 - PED["y0"]))
            draw_person(img, PED["x"], py)
            ped_dets.append({"id": 99, "cls": "person",
                             "bbox": [PED["x"], py, PED["x"] + 24, py + 56],
                             "conf": 1.0})

        # ---- ground truth --------------------------------------------------
        rec = {"frame": fi, "t": round(t, 3), "dets": []}
        for c in cars:
            if c.dead or c.x < -100 or c.x > W + 100:
                continue
            rec["dets"].append({"id": c.tid, "cls": "car",
                                "bbox": [round(c.x), c.y, round(c.x + 96), c.y + 44],
                                "conf": 1.0})
        rec["dets"] += ped_dets
        gt_f.write(json.dumps(rec) + "\n")

        cv2.putText(img, f"synthetic demo  t={t:.1f}s", (16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(img)

    vw.release()
    gt_f.close()
    print(f"wrote {out_path} ({n_frames} frames) and {args.gt}")


if __name__ == "__main__":
    main()
