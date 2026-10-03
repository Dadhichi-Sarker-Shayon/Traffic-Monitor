"""Score the event engine against labelled clips.

usage: score3.py DETS_DIR GROUND_TRUTH.json TAG [--rules-only-print]
Replays dumped YOLO detections through EventEngine and compares with labels.
"""
import sys, json, os

sys.path.insert(0, r"G:\My Drive\Relate anything\realtime-video-tracking")
from traffic_watch.events import EngineConfig, EventEngine

DETDIR, GTFILE, TAG = sys.argv[1], sys.argv[2], sys.argv[3]
GT = json.load(open(GTFILE))
TYPES = {"ACCIDENT": "accident", "JAM": "jam", "STOPPED": "stopped", "PED_CONFLICT": "ped"}

res = {}
for name, gt in GT.items():
    if name.startswith("_"):
        continue
    p = os.path.join(DETDIR, name + ".jsonl")
    if not os.path.exists(p):
        continue
    meta = json.load(open(p.replace(".jsonl", ".meta.json")))
    eng = EventEngine(EngineConfig(), frame_shape=(meta["h"], meta["w"]))
    evs, last_t = [], 0
    for line in open(p):
        r = json.loads(line)
        last_t = r["t"]
        evs += eng.update(r["t"], r["dets"])
    evs += eng.finalize(last_t)
    res[name] = {
        "dur": last_t,
        "events": [(e.type, round(e.t, 1), e.detail[:90]) for e in evs],
        "boxes": [(e.type, round(e.t, 1), list(e.bbox) if e.bbox else None) for e in evs],
    }
json.dump(res, open(os.path.join(r"D:\traffic_eval", f"engine_events_{TAG}.json"), "w"), indent=1)


def fmt(v):
    return "  n/a" if v is None else f"{v:5.2f}"


def score(subset, label):
    out = {}
    for et, key in TYPES.items():
        tp = fp = fn = tn = timed_ok = timed_n = 0
        for name, r in res.items():
            g = GT[name]
            if g.get(key) is None or not subset(g):
                continue
            fired = [e for e in r["events"] if e[0] == et]
            pred = bool(fired)
            if g[key]:
                tp += pred
                fn += (not pred)
                if key == "accident" and pred:
                    timed_n += 1
                    timed_ok += any(w[0] <= e[1] <= w[1] + 6 for e in fired for w in g["windows"])
            else:
                fp += pred
                tn += (not pred)
        pr = tp / (tp + fp) if tp + fp else None
        rc = tp / (tp + fn) if tp + fn else None
        f1 = 2 * pr * rc / (pr + rc) if pr and rc else (0.0 if (tp + fp and tp + fn) else None)
        out[et] = dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=pr, recall=rc, f1=f1,
                       timed_ok=timed_ok, timed_n=timed_n)
    print(f"\n=== {label}")
    for et, o in out.items():
        extra = f"  correct-time {o['timed_ok']}/{o['timed_n']}" if et == "ACCIDENT" else ""
        print(f"{et:13s} TP={o['tp']} FP={o['fp']} FN={o['fn']} TN={o['tn']}  "
              f"P={fmt(o['precision'])} R={fmt(o['recall'])} F1={fmt(o['f1'])}{extra}")
    return out


allr = {
    "all": score(lambda g: True, f"[{TAG}] ALL CLIPS"),
    "static": score(lambda g: g["camera"] == "static", f"[{TAG}] STATIC CAMERA"),
    "moving": score(lambda g: g["camera"] == "moving", f"[{TAG}] MOVING DASHCAM"),
    "interior": score(lambda g: g["camera"] == "interior", f"[{TAG}] INTERIOR / CAB"),
}
json.dump(allr, open(os.path.join(r"D:\traffic_eval", f"engine_scores_{TAG}.json"), "w"), indent=1)
if "--clips" in sys.argv:
    print("\n--- per clip")
    for name, r in res.items():
        g = GT[name]
        truth = [k for k in ("accident", "jam", "stopped", "ped") if g.get(k)]
        print(f"{name:24s} [{g['camera'][:4]}] truth={truth or '-'} fired={[(t, tm) for t, tm, _ in r['events']]}")
