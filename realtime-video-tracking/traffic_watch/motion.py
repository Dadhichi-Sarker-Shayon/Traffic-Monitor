"""Per-vehicle pixel motion, measured inside the detector's own box.

Why this exists: the event engine measures speed as centroid displacement, but
that is not motion. A parked car's box jitters a few pixels per frame and reads
as "moving"; a car driving towards the camera moves very few pixels and reads as
"stopped". Both mistakes land straight on the jam rule, which is defined as
"lots of cars standing still", so the engine needs to know whether the pixels
inside a box are actually changing.

For every tracked box we take the mean absolute difference between the previous
and the current grayscale crop. That is a crude motion signal, but it is the
right question: the metal is not moving if the pixels are not changing, no
matter what the box coordinates did. It is also nearly free - a handful of small
crops per frame.

Output is a mean absolute difference in [0, 1] per detection, attached as the
`motion` key so `EventEngine.update` can use it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np


class MotionMeter:
    """Rolling grayscale buffer; call `attach` once per frame."""

    def __init__(self, min_side: int = 8):
        self.min_side = min_side
        self._prev: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev = None

    def attach(self, image: np.ndarray, dets: Sequence[Dict]) -> None:
        """Attach `motion` to each detection dict, in place."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        prev = self._prev
        self._prev = gray
        if prev is None or prev.shape != gray.shape:
            for d in dets:
                d["motion"] = None
            return

        h, w = gray.shape[:2]
        for d in dets:
            d["motion"] = self._crop_motion(prev, gray, d.get("bbox"), w, h)

    def _crop_motion(self, prev: np.ndarray, cur: np.ndarray,
                     bbox, w: int, h: int) -> Optional[float]:
        if not bbox:
            return None
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, max(x1 + 1, x2)), min(h, max(y1 + 1, y2))
        if (x2 - x1) < self.min_side or (y2 - y1) < self.min_side:
            return None
        a = prev[y1:y2, x1:x2]
        b = cur[y1:y2, x1:x2]
        return float(cv2.absdiff(a, b).mean() / 255.0)