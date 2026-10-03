"""Optional VLM adjudication: an independent second opinion on events.

The heuristic engine *proposes* events from tracks and geometry; a
vision-language model looks at the pixels of the event region and
answers a fixed yes/no question. Adjudication is a verification layer,
never a trigger: it can only veto an event the heuristics already
fired, so a broken or absent model degrades to the previous behaviour
(passthrough) instead of inventing or silently dropping incidents.

A vetoed event is withdrawn, and the engine may fire it again after
its normal cooldown - so the model gets a second chance on the next
frames rather than suppressing an incident for the whole clip.

Run with --vlm. The default model is Qwen2-VL-2B (4-bit, ~1.5 GB of VRAM),
run in-process when `transformers` is importable or in a separate interpreter
with --vlm-python (needed next to OpenPSG's torch 1.13 stack). Any other model
id falls back to the transformers image-to-text pipeline.

With --vlm-scout the model also looks at a whole frame every few seconds and
may *propose* a jam or a crash on its own. That is the one place it can create
an event, so it needs two consecutive yes answers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# fixed, answerable questions per event type
QUESTIONS = {
    "ACCIDENT": "Do these two vehicles appear to have collided - are they "
                "in contact or visibly damaged? Answer yes or no.",
    "JAM": "Is this stretch of road blocked or nearly blocked by queued "
           "vehicles? Answer yes or no.",
    "JAM_ORIGIN": "Is this stretch of road blocked or nearly blocked by "
                  "queued vehicles? Answer yes or no.",
    "JAM_FRONT": "Is traffic queued up to here, with flow resuming ahead? "
                 "Answer yes or no.",
    "STOPPED": "Is this vehicle stopped while other traffic keeps moving "
               "around it? Answer yes or no.",
    "PED_CONFLICT": "Is a pedestrian in the roadway with vehicles moving "
                    "through it? Answer yes or no.",
}

# scene-level questions for the scout (whole frame, no event needed)
SCENE_QUESTIONS = {
    "ACCIDENT": "Is a vehicle crash or collision happening or has one just "
                "happened in this image? Answer yes or no.",
    "JAM": "Is this road heavily congested, with many vehicles queued closely "
           "and barely moving? Answer yes or no.",
}

# events judged on the whole frame: a tight crop of two distant cars hides the
# context (road, queue, wreckage) the model needs
FULL_FRAME_TYPES = frozenset({"ACCIDENT", "JAM", "JAM_ORIGIN", "JAM_FRONT"})

YES_WORDS = ("yes", "yeah", "yep", "y")
NO_WORDS = ("no", "nope", "nah", "n", "none", "nothing")


@dataclass
class Verdict:
    agrees: bool                     # True = report the event
    note: str = ""                   # model answer, or why we could not ask
    passthrough: bool = False        # no opinion was actually obtained


class VLMAdjudicator:
    """Second opinion for fired events. Never raises, never invents."""

    def __init__(self, model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
                 device: Optional[str] = None,
                 questions: Optional[dict] = None,
                 python_exe: Optional[str] = None,
                 adapter: Optional[str] = None,
                 full_frame_types=FULL_FRAME_TYPES):
        self.model_id = model_id
        self.device = device
        self.python_exe = python_exe
        self.adapter = adapter
        self.full_frame_types = frozenset(full_frame_types)
        self.questions = questions or QUESTIONS
        self._pipe = None
        self.last_error: Optional[str] = None
        self.calls = 0
        self.agreements = 0
        self.vetoes = 0
        self.passthroughs = 0

    # ---------------------------------------------------------- model loading
    def _ensure_pipe(self) -> bool:
        if self._pipe is not None:
            return True
        try:
            dev = self.device or "cuda"
            if self.python_exe:
                from .vlm_backend import WorkerPipe
                try:
                    self._pipe = WorkerPipe(self.python_exe, self.model_id, dev, adapter=self.adapter)
                except Exception:  # one retry: a cold start can lose a race for RAM/VRAM
                    self._pipe = WorkerPipe(self.python_exe, self.model_id, dev, adapter=self.adapter)
            elif "qwen2" in self.model_id.lower() and "vl" in self.model_id.lower():
                from .vlm_backend import QwenVL
                self._pipe = QwenVL(self.model_id, dev, adapter=self.adapter)
            else:
                from transformers import pipeline
                kw = {"model": self.model_id}
                if self.device is not None:
                    kw["device"] = self.device
                self._pipe = pipeline("image-to-text", **kw)
            return True
        except Exception as e:  # no torch / model / network: passthrough
            self.last_error = f"{type(e).__name__}: {e}"
            return False

    def close(self) -> None:
        closer = getattr(self._pipe, "close", None)
        if closer:
            closer()

    # ------------------------------------------------------------ inference
    def _ask(self, image_rgb, question: str) -> str:
        out = self._pipe(image_rgb, prompt=question)
        if isinstance(out, list):
            out = out[0] if out else {}
        if isinstance(out, dict):
            return str(out.get("generated_text", ""))
        return str(out)

    # ------------------------------------------------------------ public api
    def verify(self, image_bgr, event) -> Verdict:
        """Check one fired event against the pixels. Never raises."""
        question = self.questions.get(event.type)
        if question is None:
            return Verdict(True, f"no question for {event.type}",
                           passthrough=True)
        if not self._ensure_pipe():
            self.passthroughs += 1
            return Verdict(True, f"vlm unavailable ({self.last_error})",
                           passthrough=True)
        if event.type in self.full_frame_types:
            crop = np.ascontiguousarray(image_bgr[:, :, ::-1])
        else:
            crop = crop_event(image_bgr, event)
        self.calls += 1
        try:
            answer = self._ask(crop, question)
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            self.passthroughs += 1
            return Verdict(True, f"vlm error ({self.last_error})",
                           passthrough=True)
        verdict = _parse_answer(answer)
        if verdict is None:
            # an unparseable answer cannot confirm anything, but it must
            # not silently drop a real incident either
            self.passthroughs += 1
            return Verdict(True, f"unparseable answer: {answer!r}",
                           passthrough=True)
        if verdict:
            self.agreements += 1
            return Verdict(True, answer.strip())
        self.vetoes += 1
        return Verdict(False, answer.strip())


    def ask_scene(self, image_bgr, kind: str) -> Optional[bool]:
        """Whole-frame yes/no for the scout; None when no answer was obtained."""
        q = SCENE_QUESTIONS.get(kind)
        if q is None or not self._ensure_pipe():
            return None
        self.calls += 1
        try:
            answer = self._ask(np.ascontiguousarray(image_bgr[:, :, ::-1]), q)
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            return None
        return _parse_answer(answer)


class VLMScout:
    """Lets the VLM propose jams and crashes by looking at sampled frames.

    Samples one frame every `interval_s` of video, asks the scene questions,
    and reports a kind only after `need` consecutive yes answers, then stays
    quiet for `cooldown_s`. The engine's own events are never duplicated:
    `skip` is checked first (e.g. an already-active JAM).
    """

    def __init__(self, adjudicator: "VLMAdjudicator", interval_s: float = 4.0,
                 need: int = 2, cooldown_s: float = 20.0,
                 kinds=("ACCIDENT", "JAM")):
        self.adj = adjudicator
        self.interval_s = interval_s
        self.need = need
        self.cooldown_s = cooldown_s
        self.kinds = tuple(kinds)
        self._last_t = -1e9
        self._streak = {k: 0 for k in self.kinds}
        self._last_fired = {k: -1e9 for k in self.kinds}

    def scan(self, image_bgr, t: float, skip=()) -> list:
        from .events import Event

        if t - self._last_t < self.interval_s:
            return []
        self._last_t = t
        out = []
        for kind in self.kinds:
            if kind in skip:
                self._streak[kind] = 0
                continue
            ans = self.adj.ask_scene(image_bgr, kind)
            if ans is None:
                continue
            self._streak[kind] = self._streak[kind] + 1 if ans else 0
            if (self._streak[kind] >= self.need
                    and t - self._last_fired[kind] >= self.cooldown_s):
                self._last_fired[kind] = t
                out.append(Event(
                    type=kind, t=t, key=f"VLM_{kind}",
                    detail=f"vlm scene check: {self._streak[kind]} consecutive yes",
                    evidence={"source": "vlm", "consecutive_yes": self._streak[kind]}))
        return out


def crop_event(image_bgr, event, pad_frac: float = 0.15):
    """BGR crop around the event bbox (or the whole frame), as RGB."""
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = (event.bbox or (0, 0, w, h))
    pw, ph = (x2 - x1) * pad_frac, (y2 - y1) * pad_frac
    x1, y1 = int(max(0, x1 - pw)), int(max(0, y1 - ph))
    x2, y2 = int(min(w, x2 + pw)), int(min(h, y2 + ph))
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        crop = image_bgr
    return np.ascontiguousarray(crop[:, :, ::-1])  # BGR -> RGB


def _parse_answer(text: str) -> Optional[bool]:
    t = text.strip().lower()
    if not t:
        return None
    first = t.split()[0].strip(".,!?")
    if first in YES_WORDS:
        return True
    if first in NO_WORDS:
        return False
    if "yes" in t[:20]:
        return True
    if "no" in t[:20]:
        return False
    return None
