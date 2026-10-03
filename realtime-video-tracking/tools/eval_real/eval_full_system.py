"""Full-system evaluation: rules / rules+VLM veto / rules+veto+VLM scout.

usage: eval_full.py DETS_DIR CLIPS_DIR GROUND_TRUTH.json TAG PYTHON_EXE [scout_interval_s]
Replays dumped YOLO detections through EventEngine; the VLM (the project's own
VLMAdjudicator + worker process) judges each fired event on a clean frame and the
scout samples whole frames. Frames are read from the video by timestamp.
"""
import sys, json, os, time

sys.path.insert(0, r"G:\My Drive\Relate anything\realtime-video-tracking")
import cv2
from traffic_watch.events import EngineConfig, EventEngine, Event
from traffic_watch.adjudicator import VLMAdjudicator, VLMScout

DETDIR, CLIPDIR, GTFILE, TAG, PYEXE = sys.argv[1:6]
INTERVAL = float(sys.argv[6]) if len(sys.argv) > 6 else 6.0
GT = json.load(open(GTFILE))
ADJUDICATE = {"ACCIDENT", "JAM", "STOPPED", "PED_CONFLICT"}
OUT = os.path.join(r"D:\traffic_eval", f"full_{TAG}.json")
res = json.load(open(OUT)) if os.path.exists(OUT) else {}

adj = VLMAdjudicator(python_exe=PYEXE, device="cuda")


def frame_at(clip, t):
    cap = cv2.VideoCapture(os.path.join(CLIPDIR, clip + ".mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(t * fps)))
    ok, f = cap.read()
    cap.release()
    return f if ok else None


t0 = time.time()
for name in GT:
    if name.startswith("_") or name in res:
        continue
    p = os.path.join(DETDIR, name + ".jsonl")
    if not os.path.exists(p):
        continue
    meta = json.load(open(p.replace(".jsonl", ".meta.json")))
    eng = EventEngine(EngineConfig(), frame_shape=(meta["h"], meta["w"]))
    evs, last_t, persons = [], 0, 0
    for line in open(p):
        r = json.loads(line)
        last_t = r["t"]
        persons += sum(1 for d in r["dets"] if d["cls"] == "person")
        evs += eng.update(r["t"], r["dets"])
    evs += eng.finalize(last_t)

    rows = []
    for e in evs:
        if e.type not in ADJUDICATE:
            continue
        v = None
        f = frame_at(name, e.t)
        if f is not None:
            v = adj.verify(f, e)
        rows.append({"type": e.type, "t": round(e.t, 1), "detail": e.detail[:80],
                     "kept": True if v is None else bool(v.agrees),
                     "passthrough": bool(v.passthrough) if v else True,
                     "vlm": v.note if v else "no frame"})
    # scout: sample frames; skip kinds the (surviving) engine events already cover
    scout = VLMScout(adj, interval_s=INTERVAL, need=2, cooldown_s=20.0)
    scouted = []
    t = INTERVAL / 2
    while t < last_t:
        f = frame_at(name, t)
        if f is not None:
            skip = set()
            for r_ in rows:
                if r_["kept"] and r_["t"] <= t and r_["type"] in ("JAM",):
                    skip.add("JAM")
                if r_["kept"] and r_["type"] == "ACCIDENT" and 0 <= t - r_["t"] < 20:
                    skip.add("ACCIDENT")
            for ev in scout.scan(f, t, skip=skip):
                scouted.append({"type": ev.type, "t": round(ev.t, 1)})
        t += INTERVAL
    res[name] = {"dur": last_t, "persons": persons, "engine": rows, "scout": scouted,
                 "vlm_errors": adj.last_error}
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"{name:24s} engine={[(r['type'], r['t'], 'KEEP' if r['kept'] else 'veto') for r in rows]} "
          f"scout={[(s['type'], s['t']) for s in scouted]}  [{time.time()-t0:.0f}s]", flush=True)
adj.close()
print("DONE", round(time.time() - t0), "s", flush=True)
