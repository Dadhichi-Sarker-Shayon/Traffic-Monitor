"""Video source abstraction: video file, webcam index, or RTSP/HTTP URL."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Optional, Union

import cv2
import numpy as np

SourceSpec = Union[str, int]


@dataclass
class Frame:
    index: int
    t: float  # seconds, from source clock (wall clock for live sources)
    image: np.ndarray  # BGR


class VideoSource:
    """Iterates frames of a file, webcam, or stream.

    For files, ``t`` is the media timestamp (index / fps) so playback speed
    does not affect event thresholds.  For live sources we use the wall clock.
    """

    def __init__(self, spec: SourceSpec, *, max_frames: Optional[int] = None):
        self.spec = spec
        self.max_frames = max_frames
        self.live = isinstance(spec, int) or (
            isinstance(spec, str)
            and (spec.startswith("rtsp://") or spec.startswith("http"))
        )
        cap = cv2.VideoCapture(spec)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {spec!r}")
        self._cap = cap
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if not (1.0 <= self.fps <= 240.0):
            self.fps = 30.0
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._index = 0
        self._wall0 = time.monotonic()

    def frames(self) -> Iterator[Frame]:
        while True:
            if self.max_frames is not None and self._index >= self.max_frames:
                return
            ok, image = self._cap.read()
            if not ok:
                return
            if self.live:
                t = time.monotonic() - self._wall0
            else:
                t = self._index / self.fps
            yield Frame(index=self._index, t=t, image=image)
            self._index += 1

    def close(self) -> None:
        try:
            self._cap.release()
        except Exception:
            pass

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def open_source(spec: SourceSpec, *, max_frames: Optional[int] = None) -> VideoSource:
    """``spec`` may be a path, an int webcam index, or an rtsp/http URL."""
    if isinstance(spec, str) and spec.isdigit():
        spec = int(spec)
    return VideoSource(spec, max_frames=max_frames)
