"""Score the event engine against the labelled eval windows.

Reads eval/windows.jsonl (after labelling) plus the per-clip event logs and
reports, per rule, how many false positives and false negatives we make. CPU
only - the engine's answers are already in the JSONL logs next to each clip.

Two different questions, because the rules are two different kinds of thing:

  * ACCIDENT is an *instant*. The question is "did it fire inside this window".
  * JAM is a *state*. It fires once when the queue forms and then stays active,
    so the question is "was it active at any point during this window". Firing
    at 7.5s and holding until 34s is one positive for every window in that
    span - counting only the transition timestamp would score six of them as
    misses, which is why the state rules reconstruct active intervals from the
    log: a fire at time t is active until the next fire of the same type.

A window counts as a positive for a rule if the label says so.

    python tools/score_eval.py
    python tools/score_eval.py --report eval/report.md --verbose
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

INF = float("inf")

# (label key, rule name, engine event types, instant or state)
RULES = [
    ("label_jam", "JAM", ("JAM",), "state"),
    ("label_accident", "ACCIDENT", ("ACCIDENT",), "instant"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dir", default="eval")
    p.add_argument("--report", default=None, help="write a markdown report here")
    p.add_argument("--verbose", action="store_true", help="list every disagreement")
    return p.parse_args()


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def fire_times(log_path: str) -> Dict[str, List[float]]:
    """{event type: sorted list of times it fired} from one clip's log."""
    out: Dict[str, List[float]] = {}
    with open(log_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("kind") == "event":
                out.setdefault(str(row.get("type")), []).append(float(row.get("t", 0.0)))
    for v in out.values():
        v.sort()
    return out


def intervals(times: List[float], duration: float) -> List[Tuple[float, float]]:
    """Active spans implied by a sequence of transitions."""
    spans = []
    for i, t in enumerate(times):
        end = times[i + 1] if i + 1 < len(times) else max(duration, t + 1e-6)
        spans.append((t, end))
    return spans


class ClipLog:
    """Cached per-clip event log, keyed by the path stored in the window."""

    def __init__(self, eval_dir: str):
        self.eval_dir = eval_dir
        self._cache: Dict[str, Dict[str, List[float]]] = {}

    def times(self, row: dict) -> Dict[str, List[float]]:
        rel = row.get("log")
        if not rel:
            return {}
        path = os.path.normpath(os.path.join(self.eval_dir, rel))
        if path not in self._cache:
            self._cache[path] = fire_times(path) if os.path.exists(path) else {}
        return self._cache[path]


def detected(row: dict, types: Tuple[str, ...], kind: str, logs: ClipLog) -> Optional[float]:
    """Was the rule active in this window? Returns the time of first evidence."""
    times = logs.times(row)
    a, b = row["t_start"], row["t_end"]
    dur = row.get("duration", b)
    best: Optional[float] = None
    for t in types:
        fires = times.get(t, [])
        if kind == "instant":
            for when in fires:
                if a <= when <= b and (best is None or when < best):
                    best = when
        else:
            # a fire stays active until the next fire of the same type
            for s, e in intervals(fires, dur):
                if a < e and s < b:
                    cand = max(s, a)
                    if best is None or cand < best:
                        best = cand
    return best


def main() -> int:
    args = parse_args()
    path = os.path.join(args.eval_dir, "windows.jsonl")
    if not os.path.exists(path):
        print(f"no {path} - run tools/make_eval_set.py then tools/label_eval.py")
        return 1
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    logs = ClipLog(args.eval_dir)

    labelled = [r for r in rows if r.get("label_jam") is not None
                or r.get("label_accident") is not None]
    watchable = [r for r in labelled if r.get("label_visibility") is not False]
    real = [r for r in watchable if r.get("label_source") != "authored"]
    synthetic = [r for r in watchable if r.get("label_source") == "authored"]
    print(f"{len(rows)} windows, {len(labelled)} labelled, {len(watchable)} watchable "
          f"({len(real)} real, {len(synthetic)} synthetic)\n")

    report: List[str] = ["# Eval report", "",
                         f"{len(labelled)}/{len(rows)} windows labelled "
                         f"({len(real)} real, {len(synthetic)} synthetic).", "",
                         "Headline numbers come from real footage only; the synthetic "
                         "demo clip is reported separately because it is flat graphics, "
                         "not camera imagery.", ""]
    lines: List[str] = []

    for cohort_name, cohort in (("REAL FOOTAGE", real), ("SYNTHETIC DEMO", synthetic)):
        if not cohort:
            continue
        lines.append(f"--- {cohort_name} ({len(cohort)} windows) ---")
        report += [f"## {cohort_name} ({len(cohort)} windows)", ""]
        for key, name, types, kind in RULES:
            tp = fp = fn = 0
            fps: List[dict] = []
            fns: List[dict] = []
            for r in cohort:
                truth = bool(r.get(key))
                fired = bool(detected(r, types, kind, logs))
                if truth and fired:
                    tp += 1
                elif fired and not truth:
                    fp += 1
                    fps.append(r)
                elif truth and not fired:
                    fn += 1
                    fns.append(r)
            p, rc, f1 = prf(tp, fp, fn)
            lines.append(f"{name:9s} ({kind:7s}) tp={tp:3d} fp={fp:3d} fn={fn:3d}  "
                         f"precision={p:.2f} recall={rc:.2f} f1={f1:.2f}")
            report += [f"### {name} ({kind})", "",
                       f"tp={tp} fp={fp} fn={fn} | precision={p:.2f} "
                       f"recall={rc:.2f} f1={f1:.2f}", ""]
            if fps:
                report.append("False positives (engine says yes, label says no):")
                for r in fps:
                    report.append(f"- `{r['id']}` — chosen: {r['selected_because']}")
                report.append("")
            if fns:
                report.append("Missed (label says yes, engine silent):")
                for r in fns:
                    report.append(f"- `{r['id']}` — chosen: {r['selected_because']}")
                report.append("")
            if args.verbose:
                for title, group in (("FP", fps), ("FN", fns)):
                    for r in group:
                        lines.append(f"    {title} {r['id']} clip={r['clip']} "
                                     f"why={r['selected_because']} "
                                     f"engine={r.get('engine_events')}")
        lines.append("")

    dmg = [r for r in watchable if r.get("label_damage_visible") is True]
    jam_pos = [r for r in watchable if r.get("label_jam") is True]
    acc_pos = [r for r in watchable if r.get("label_accident") is True]
    unlabelled = len(rows) - len(labelled)
    lines.append(f"ground truth (labelled windows): {len(jam_pos)} jam, "
                 f"{len(acc_pos)} accident, {len(dmg)} with visible damage")
    if unlabelled:
        lines.append(f"{unlabelled} windows still unlabelled - these numbers are "
                     f"provisional until tools/label_eval.py is finished")
    report += ["## Ground truth", "",
               f"- jam windows: {len(jam_pos)}",
               f"- accident windows: {len(acc_pos)}",
               f"- windows with visible damage: {len(dmg)}",
               f"- still unlabelled: {unlabelled}", ""]

    print("\n".join(lines))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write("\n".join(report) + "\n")
        print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())