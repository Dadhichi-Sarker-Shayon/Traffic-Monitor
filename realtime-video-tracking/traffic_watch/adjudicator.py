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

Run with --vlm. Any instruction-tuned VLM that works with the
transformers image-to-text pipeline can be used (--vlm-model).
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

YES_WORDS = ("yes", "yeah", "yep", "y")
NO_WORDS = ("no", "nope", "nah", "n", "none", "nothing")


@dataclass
class Verdict:
    agrees: bool                     # True = report the event
    note: str = ""                   # model answer, or why we could not ask
    passthrough: bool = False        # no opinion was actually obtained


class VLMAdjudicator:
    """Second opinion for fired events. Never raises, never invents."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 device: Optional[str] = None,
                 questions: Optional[dict] = None):
        self.model_id = model_id
        self.device = device
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
            from transformers import pipeline
            kw = {"model": self.model_id}
            if self.device is not None:
                kw["device"] = self.device
            self._pipe = pipeline("image-to-text", **kw)
            return True
        except Exception as e:  # no torch / model / network: passthrough
            self.last_error = f"{type(e).__name__}: {e}"
            return False

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
