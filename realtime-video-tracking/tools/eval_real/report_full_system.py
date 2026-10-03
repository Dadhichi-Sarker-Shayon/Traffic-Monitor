"""Score the three configurations from full_<tag>.json. usage: report_full.py GT.json TAG"""
import sys, json, os

GT = json.load(open(sys.argv[1]))
TAG = sys.argv[2]
R = json.load(open(os.path.join(r"D:\traffic_eval", f"full_{TAG}.json")))
TYPES = {"ACCIDENT": "accident", "JAM": "jam", "STOPPED": "stopped", "PED_CONFLICT": "ped"}
PED_NEG_MAX_PERSONS = 60   # a clip only counts as "no pedestrian conflict" if people are barely present


def fired(name, et, mode):
    r = R[name]
    ev = [e for e in r["engine"] if e["type"] == et and (mode == "rules" or e["kept"])]
    sc = [s for s in r["scout"] if s["type"] == et] if mode == "scout" else []
    return [e["t"] for e in ev] + [s["t"] for s in sc]


def counts(mode, subset):
    out = {}
    for et, key in TYPES.items():
        tp = fp = fn = tn = tok = tn_ = 0
        for name, r in R.items():
            g = GT[name]
            if g.get(key) is None or not subset(g):
                continue
            if key == "ped" and not g[key] and r["persons"] > PED_NEG_MAX_PERSONS:
                continue
            f = fired(name, et, mode)
            pred = bool(f)
            if g[key]:
                tp += pred
                fn += (not pred)
                if key == "accident" and pred:
                    tn_ += 1
                    tok += any(w[0] <= t <= w[1] + 6 for t in f for w in g["windows"])
            else:
                fp += pred
                tn += (not pred)
        out[et] = (tp, fp, fn, tn, tok, tn_)
    return out


def line(et, c):
    tp, fp, fn, tn, tok, tn_ = c
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = 2 * p * r / (p + r) if p and r else (0.0 if (tp + fp and tp + fn) else None)
    s = lambda v: " n/a" if v is None else f"{v:4.2f}"
    return f"{et:13s} TP={tp} FP={fp} FN={fn} TN={tn}  P={s(p)} R={s(r)} F1={s(f1)}" + (f"  right-time {tok}/{tn_}" if et == "ACCIDENT" else "")


for label, sub in (("ALL", lambda g: True), ("STATIC CAMERA", lambda g: g["camera"] == "static"),
                   ("MOVING DASHCAM", lambda g: g["camera"] == "moving")):
    print(f"\n######## {TAG}: {label}")
    for mode, mname in (("rules", "RULES ONLY"), ("veto", "RULES + VLM VETO"), ("scout", "RULES + VETO + VLM SCOUT")):
        c = counts(mode, sub)
        print(f"-- {mname}")
        for et in TYPES:
            print("   " + line(et, c[et]))
vetoed = [(n, e["type"], e["t"]) for n, r in R.items() for e in r["engine"] if not e["kept"]]
print("\nVLM vetoes:", len(vetoed), "of", sum(len(r["engine"]) for r in R.values()), "adjudicated events")
