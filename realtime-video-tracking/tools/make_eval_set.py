"""Build a labelling set for measuring event-rule precision and recall.

For each clip we pick windows of `--window` seconds and save a contact sheet of
six frames spanning the window, so a human can label a whole window in one
look. Windows are stratified on purpose:

  * one window centred on every event the current engine fired -> these measure
    precision (how much of what we report is wrong),
  * evenly spaced windows over the rest -> these measure recall (what we miss),
  * negatives-only clips (the aerial ones) get a couple of windows as a sanity
    check that we do not invent events there.

Output:
  eval/windows.jsonl      one record per window, to be labelled
  eval/sheets/<id>.jpg    contact sheet per window

Nothing here needs a GPU: frames come from the clips with OpenCV and the events
come from the JSONL logs already sitting in outputs/.

    python tools/make_eval_set.py
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

CLIPS_DIR = os.path.join("sample_media", "clips")
OUTPUTS_DIR = "outputs"
SHEETS = 6  # frames per contact sheet
SHEET_W = 460  # per-tile width

# Authored ground truth for the synthetic clip, straight from the constants in
# tools/make_demo_video.py (DUR=40, CRASH_T=4.0, TOW_T=34.0, PED 22-30s). Note the
# three spans are deliberately different: the crash is an *instant* at 4s, while
# the wreck stays on the road and the queue behind it persists until the tow at
# 34s. Calling every window from 4-34s an "accident" would score the engine for
# events that already happened, so accident and damage are labelled separately.
AUTHORED_GT = {
    "demo_traffic": {
        "accident_at": 4.0,             # instant: the collision itself
        "jam_window": (4.0, 34.0),       # state: queue behind the crash
        "damage_window": (4.0, 34.0),    # state: wreck visible on the road
    },
}


def overlap(a: float, b: float, window: Tuple[float, float]) -> bool:
    """Does the window [a, b) overlap the state span?"""
    return a < window[1] and window[0] < b


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clips-dir", default=CLIPS_DIR)
    p.add_argument("--outputs-dir", default=OUTPUTS_DIR)
    p.add_argument("--eval-dir", default="eval")
    p.add_argument("--window", type=float, default=6.0, help="window length in seconds")
    p.add_argument("--silent-windows", type=int, default=8,
                   help="evenly spaced windows per clip, away from fired events")
    p.add_argument("--fps-sample", type=float, default=6.0,
                   help="target sample rate of the emitted clips")
    p.add_argument("--clips", nargs="*", default=None, help="clip stems (default: all)")
    return p.parse_args()


def find_log(outputs_dir: str, stem: str) -> Optional[str]:
    """Map a clip stem to the event log we produced for it."""
    for cand in (f"{stem}_full.events.jsonl", "demo_nopsg.events.jsonl"):
        path = os.path.join(outputs_dir, cand)
        if os.path.exists(path):
            return path
    return None


def read_events(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("kind") == "event":
                out.append(row)
    return out


def contact_sheet(cap: "cv2.VideoCapture", start: int, end: int,
                  width: int, height: int) -> Optional["np.ndarray"]:
    """Tile frames sampled evenly across [start, end) into one image."""
    span = max(1, end - start)
    idxs = [start + int(span * (k + 0.5) / SHEETS) for k in range(SHEETS)]
    tiles = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        scale = SHEET_W / float(w)
        tile = cv2.resize(frame, (SHEET_W, max(1, int(h * scale))))
        cv2.putText(tile, f"f{i}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(tile, f"f{i}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return None
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    rows = [r for r in rows if r.shape[1] == max(x.shape[1] for x in rows)]
    return np.vstack(rows)


def main() -> int:
    args = parse_args()
    os.makedirs(os.path.join(args.eval_dir, "sheets"), exist_ok=True)
    os.makedirs(os.path.join(args.eval_dir, "samples"), exist_ok=True)

    stems = args.clips or sorted(
        os.path.splitext(f)[0] for f in os.listdir(args.clips_dir) if f.endswith(".mp4"))

    records: List[dict] = []
    for stem in stems:
        clip = os.path.join(args.clips_dir, f"{stem}.mp4")
        if not os.path.exists(clip):
            print(f"[skip] {clip} missing")
            continue
        cap = cv2.VideoCapture(clip)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if n_frames <= 0:
            print(f"[skip] {stem}: unreadable")
            cap.release()
            continue

        log = find_log(args.outputs_dir, stem)
        events = read_events(log) if log else []
        win_frames = int(round(args.window * fps))

        # keep windows inside the clip
        def clamp(t: float) -> int:
            return int(max(0, min(n_frames - win_frames, round(t * fps) - win_frames // 2)))

        starts: Dict[int, str] = {}  # start_frame -> why it was chosen

        # 1) one window per fired event -> precision
        for ev in events:
            s = clamp(float(ev.get("t", 0.0)))
            starts.setdefault(s, f"event:{ev.get('type')}")

        # 2) evenly spaced windows away from the fired ones -> recall
        step = max(1, n_frames // max(1, args.silent_windows))
        for s in range(0, max(1, n_frames - win_frames), step):
            cand = int(s)
            if any(abs(cand - other) < win_frames for other in starts):
                continue
            if starts.get(cand):
                continue
            starts[cand] = "silent"

        for start, why in sorted(starts.items()):
            end = min(n_frames, start + win_frames)
            sheet = contact_sheet(cap, start, end, w, h)
            if sheet is None:
                continue
            wid = f"{stem}_{start:06d}"
            sheet_path = os.path.join(args.eval_dir, "sheets", f"{wid}.jpg")
            cv2.imwrite(sheet_path, sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 82])

            # a small sampled copy of the window, so the eval set is self-contained
            samp_dir = os.path.join(args.eval_dir, "samples")
            cv2.imwrite(os.path.join(samp_dir, f"{wid}.jpg"), sheet,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 70])

            records.append({
                "id": wid,
                "clip": stem,
                "start_frame": start,
                "end_frame": end,
                "t_start": round(start / fps, 3),
                "t_end": round(end / fps, 3),
                "n_frames": n_frames,
                "fps": round(fps, 3),
                "frame_size": [w, h],
                "sheet": os.path.join("sheets", f"{wid}.jpg"),
                "log": os.path.relpath(log, args.eval_dir) if log else None,
                "duration": round(n_frames / fps, 3),
                "selected_because": why,
                "engine_events": [e.get("type") for e in events
                                  if start / fps - 0.5 <= float(e.get("t", 0.0)) <= end / fps + 0.5],
                # labels to fill in (see tools/label_eval.py):
                "label_jam": None,          # True | False | None(unsure)
                "label_accident": None,     # True | False | None(unsure)
                "label_damage_visible": None,
                "label_visibility": None,   # None if the window is watchable
                "label_source": "human",    # human | authored
                "note": "",
            })
        cap.release()
        print(f"[{stem}] {n_frames}f @{fps:.2f}fps, log={os.path.basename(log) if log else 'none'}, "
              f"{len(events)} events -> {len(starts)} windows")

    # fill in what the generator of the synthetic clip already knows
    authored = 0
    for r in records:
        gt = AUTHORED_GT.get(r["clip"])
        if not gt:
            continue
        span = (r["t_start"], r["t_end"])
        r["label_jam"] = overlap(*span, gt["jam_window"])
        r["label_damage_visible"] = overlap(*span, gt["damage_window"])
        # an instant belongs to the window it happens in, not to every window
        # that touches it
        crash = gt.get("accident_at")
        r["label_accident"] = (r["t_start"] <= crash <= r["t_end"]) if crash is not None else None
        r["label_visibility"] = True
        r["label_source"] = "authored"
        r["note"] = "auto-labelled from make_demo_video.py constants"
        authored += 1
    if authored:
        print(f"auto-labelled {authored} synthetic windows from the generator")

    path = os.path.join(args.eval_dir, "windows.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(records)} windows -> {path}")
    print("next: python tools/label_eval.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())