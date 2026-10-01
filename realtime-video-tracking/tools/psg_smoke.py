"""Smoke test for the OpenPSG adapter: one frame in -> scene graph out."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from traffic_watch.psg import PSGRunner  # noqa: E402


def main() -> int:
    video = ROOT / "sample_media" / "clips" / "demo_traffic.mp4"
    if not video.exists():
        print(f"missing {video}; run tools/make_demo_video.py first")
        return 1

    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 300)  # t=10s: crash + queue present
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("cannot read frame")
        return 1

    runner = PSGRunner(interval=0)  # manual mode, no thread
    t0 = time.perf_counter()
    try:
        res = runner.infer(frame, t=10.0, frame_index=300)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"PSG FAILED: {type(e).__name__}: {e}")
        return 1
    dt = time.perf_counter() - t0

    print(f"PSG OK in {dt*1000:.0f} ms  device={runner.device}")
    print(f"instances ({len(res.instances)}):")
    for inst in res.instances[:20]:
        print(f"  - {inst.label:24s} area={inst.area:7d} bbox={tuple(round(v) for v in inst.bbox)}")
    print(f"relations ({len(res.relations)}):")
    for r in res.top_relations(12):
        s = res.instances[r.s].label
        o = res.instances[r.o].label
        print(f"  - {s} -[{r.predicate} {r.score:.2f}]-> {o}")
    print(f"stuff: {sorted(res.stuff.keys())}")
    print(f"sentence: {res.sentence()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
