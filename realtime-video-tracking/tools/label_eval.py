"""Local GUI for labelling the eval windows produced by make_eval_set.py.

Keyboard-only on purpose: labelling 40-60 windows by hand is the bottleneck of
this whole exercise, so every decision is one keypress.

    Left / Right   select field (jam | accident | damage | visibility)
    y / n / u      yes / no / unsure for the selected field
    Enter          save and go to the next unlabelled window
    Backspace      go to the previous window
    p              toggle "skip this window" (jumps to the next unlabelled)
    t              type a free-text note
    q              save everything and quit

Everything stays on the machine; nothing is uploaded.
"""

from __future__ import annotations

import argparse
import json
import os
import tkinter as tk
from tkinter import simpledialog
from typing import Dict, List

FIELDS = [
    ("label_jam", "TRAFFIC JAM (many cars standing, not moving)"),
    ("label_accident", "ACCIDENT (crash happened in this window)"),
    ("label_damage_visible", "DAMAGE VISIBLE (body damage / debris / glass)"),
    ("label_visibility", "WATCHABLE (camera angle fine enough to judge)"),
]
YES, NO, UNSURE = True, False, None
GLYPH = {True: "YES", False: "no", None: "unsure", "skip": "skip"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dir", default="eval")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--filter", default="", help="only windows whose id contains this")
    return p.parse_args()


class Labeler:
    def __init__(self, eval_dir: str, start: int, filter_: str):
        self.dir = eval_dir
        self.path = os.path.join(eval_dir, "windows.jsonl")
        self.rows: List[dict] = [json.loads(l) for l in open(self.path, encoding="utf-8") if l.strip()]
        if filter_:
            self.rows = [r for r in self.rows if filter_ in r["id"]]
        self.i = min(start, len(self.rows) - 1)
        self.field = 0

        self.root = tk.Tk()
        self.root.title("traffic_watch eval labeller")
        self.root.configure(bg="#141414")
        self.root.bind("<Key>", self.on_key)

        self.info = tk.Label(self.root, bg="#141414", fg="#dddddd",
                             font=("Consolas", 10), anchor="w", justify="left")
        self.info.pack(fill="x", padx=10, pady=(10, 4))

        self.img_label = tk.Label(self.root, bg="#141414")
        self.img_label.pack(fill="both", expand=True, padx=10)

        self.status = tk.Label(self.root, bg="#141414", fg="#eeeeee",
                               font=("Consolas", 11), anchor="w", justify="left")
        self.status.pack(fill="x", padx=10, pady=(6, 10))

        self.root.bind("<Configure>", lambda _e: self.show())
        self.show()

    # ------------------------------------------------------------------ util
    @property
    def row(self) -> dict:
        return self.rows[self.i]

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as fh:
            for r in self.rows:
                fh.write(json.dumps(r) + "\n")

    def show(self) -> None:
        r = self.row
        # cv2 -> PPM -> PhotoImage keeps this dependency-free and local
        import cv2
        from PIL import Image, ImageTk
        sheet = os.path.join(self.dir, r["sheet"])
        if os.path.exists(sheet):
            arr = cv2.imread(sheet)
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(arr)
            w = min(pil.width, self.root.winfo_width() - 20 or pil.width)
            if w < pil.width:
                pil = pil.resize((w, int(pil.height * w / pil.width)))
            self.photo = ImageTk.PhotoImage(pil)
            self.img_label.configure(image=self.photo)

        self.info.configure(text=(
            f"[{self.i + 1}/{len(self.rows)}]  {r['id']}\n"
            f"clip {r['clip']}  frames {r['start_frame']}-{r['end_frame']}  "
            f"t={r['t_start']:.1f}-{r['t_end']:.1f}s   {r['frame_size'][0]}x{r['frame_size'][1]}\n"
            f"chosen because: {r['selected_because']}   engine fired: "
            f"{', '.join(r['engine_events']) if r['engine_events'] else 'nothing'}"))

        lines = []
        for idx, (key, label) in enumerate(FIELDS):
            mark = ">>" if idx == self.field else "  "
            lines.append(f"{mark} {label:<52} {GLYPH.get(r.get(key), 'unsure')}")
        done = sum(1 for r2 in self.rows if r2.get("label_jam") is not None
                   or r2.get("label_accident") is not None)
        lines.append("")
        lines.append(f"labelled {done}/{len(self.rows)}"
                     + (f"   note: {r['note']}" if r.get("note") else ""))
        lines.append("")
        lines.append("ACCIDENT is an instant: say YES only if the crash happens inside")
        lines.append("this window - a wreck that is merely still on the road is DAMAGE,")
        lines.append("not a new accident. JAM is a state: YES if cars stand still at any")
        lines.append("point in this window. Use WATCHABLE=no for windows you cannot judge.")
        lines.append("Left/Right field | y/n/u set | Enter next | Backspace prev | p skip | t note | q quit")
        self.status.configure(text="\n".join(lines))

    # ------------------------------------------------------------------ keys
    def set_field(self, value) -> None:
        key = FIELDS[self.field][0]
        self.row[key] = value
        self.save()
        self.show()

    def goto(self, delta: int) -> None:
        self.i = max(0, min(len(self.rows) - 1, self.i + delta))
        self.show()

    def next_unlabelled(self, start: int) -> None:
        n = len(self.rows)
        for k in range(1, n + 1):
            j = (start + k) % n
            if self.rows[j].get("label_jam") is None:
                self.i = j
                self.show()
                return
        self.status.configure(text=self.status.cget("text") + "\nall windows labelled")

    def on_key(self, ev) -> None:
        k = ev.keysym
        if k in ("Left", "Right"):
            self.field = (self.field + (1 if k == "Right" else -1)) % len(FIELDS)
            self.show()
        elif k in ("y", "Y"):
            self.set_field(YES)
        elif k in ("n", "N"):
            self.set_field(NO)
        elif k in ("u", "U"):
            self.set_field(UNSURE)
        elif k == "Return":
            self.next_unlabelled(self.i)
        elif k == "BackSpace":
            self.goto(-1)
        elif k in ("p", "P"):
            self.row["note"] = (self.row.get("note", "") + " [skipped]").strip()
            self.save()
            self.next_unlabelled(self.i)
        elif k in ("t", "T"):
            self.row["note"] = simpledialog.askstring(
                "Note", "Note for this window:", initialvalue=self.row.get("note", ""))
            self.save()
            self.show()
        elif k in ("q", "Q", "Escape"):
            self.save()
            self.root.destroy()


def main() -> int:
    args = parse_args()
    if not os.path.exists(os.path.join(args.eval_dir, "windows.jsonl")):
        print("no eval set found - run tools/make_eval_set.py first")
        return 1
    lab = Labeler(args.eval_dir, args.start, args.filter)
    lab.root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())