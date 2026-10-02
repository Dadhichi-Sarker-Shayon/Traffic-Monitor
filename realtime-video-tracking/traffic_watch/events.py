"""Event engine: decides *what is happening* from tracked objects over time.

Pure logic (numpy only) so it can be unit-tested without the heavy MM stack.

Events produced
---------------
JAM            a zone whose vehicle density is high and speed collapsed
JAM_ORIGIN     the zone where the jam began (earliest onset, still jammed)
ACCIDENT       collision-like overlap of two moving vehicles, or a
               sudden multi-vehicle speed collapse in one zone
STOPPED        a vehicle standing for a long time inside an *active* flow
PED_CONFLICT   a pedestrian inside the roadway among moving vehicles

All detections are scores with hysteresis + persistence so a single noisy
frame never fires an event.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle", "bicycle"}
MOVING_CLASSES = VEHICLE_CLASSES

# which EngineConfig field holds the retrospective confirmation time per event
CONFIRM_ATTR = {
    "JAM": "confirm_jam_s",
    "JAM_ORIGIN": "confirm_jam_s",
    "JAM_FRONT": "confirm_jam_s",
    "ACCIDENT": "confirm_accident_s",
    "STOPPED": "confirm_stopped_s",
    "PED_CONFLICT": "confirm_ped_s",
}


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
@dataclass
class EngineConfig:
    # zone grid (zones cover the ROI, by default the lower part of the frame)
    grid_cols: int = 6
    grid_rows: int = 3
    roi_top_frac: float = 0.45  # ROI starts at 45% of frame height

    # speed (pixels / second, EMA smoothed)
    speed_ema: float = 0.3
    jam_speed_px_s: float = 18.0        # "abs" mode: below this a zone counts as congested
    # "abs" keeps that fixed pixel threshold; "auto" scales it to the traffic
    # speed actually seen in this clip, which is the only way a px/s threshold
    # can mean the same thing at 720p and 4K, or at 30 m and 80 m up. With
    # px_per_meter set, "metric" takes the threshold in m/s instead.
    jam_speed_mode: str = "abs"        # abs | auto | metric
    jam_speed_rel_frac: float = 0.25   # auto: congested below this share of the scene speed
    jam_speed_mps: float = 1.5         # metric: ~5 km/h
    px_per_meter: Optional[float] = None
    # a car whose pixels are not changing is standing still, however much its
    # box wanders between frames
    motion_still_thresh: float = 0.012  # mean abs frame difference inside the box
    jitter_speed_px_s: float = 40.0# ... below this, displacement is treated as noise
    jam_min_vehicles: int = 3           # vehicles inside the zone
    jam_min_density: float = 0.06       # vehicles / zone area (k px)
    jam_persist_s: float = 4.0          # zone must stay bad this long
    jam_origin_margin_s: float = 2.0    # how much earlier origin must onset
    jam_front_speed_px_s: float = 45.0  # speed of a "flowing" zone near the jam

    # accident
    collision_iou: float = 0.25
    collision_min_speed: float = 60.0   # px/s before impact
    collision_persist_s: float = 0.6
    # An impact is a discontinuity: speed collapses within a few
    # frames. Overlap without such a jump is a merge or a queue
    # bumper, so the largest recent per-0.5s speed drop must clear
    # this bar before overlap reads as a collision.
    collision_dv_px_s: float = 25.0
    collapse_drop_frac: float = 0.65    # median speed drops by 65%+
    collapse_min_before: float = 55.0   # ... from at least this speed
    collapse_persist_s: float = 1.5

    # stopped vehicle
    stopped_min_s: float = 8.0
    stopped_min_speed: float = 12.0     # px/s considered "stopped"

    # pedestrian conflict
    ped_conflict_min_vehicles: int = 2
    ped_conflict_persist_s: float = 1.0

    # generic
    event_on: float = 1.0               # score needed to fire
    event_off: float = 0.45             # score below which a fired event clears
    event_max: float = 3.0              # score cap

    # Retrospective confirmation. The rules above are all "past frames only",
    # which is half the problem: a pair of cars overlapping for one frame, or a
    # queue that dissolves immediately, both look real until you see what
    # happens next. A candidate is therefore not reported the moment its score
    # crosses the threshold - it has to still be there this many seconds later,
    # and it is retracted silently if the evidence falls apart before then.
    # 0.0 restores the old fire-immediately behaviour.
    confirm_jam_s: float = 1.0
    confirm_accident_s: float = 1.5
    confirm_stopped_s: float = 0.0
    confirm_ped_s: float = 0.0
    history_s: float = 6.0              # per-track history window
    cool_down_s: float = 6.0            # min gap between repeats of same event


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(1e-6, area_a + area_b - inter)


def _median(vals: List[float], default: float = 0.0) -> float:
    return float(np.median(vals)) if vals else default


# --------------------------------------------------------------------------- #
# zones
# --------------------------------------------------------------------------- #
@dataclass
class ZoneState:
    col: int
    row: int
    vehicles: int = 0
    pedestrians: int = 0
    median_speed: float = 0.0
    jammed: bool = False
    jam_score: float = 0.0
    jam_onset_t: Optional[float] = None  # first time it became jammed

    @property
    def key(self) -> Tuple[int, int]:
        return (self.col, self.row)


@dataclass
class Event:
    type: str  # JAM | JAM_ORIGIN | ACCIDENT | STOPPED | PED_CONFLICT
    t: float
    zone: Optional[Tuple[int, int]] = None
    track_ids: Tuple[int, ...] = ()
    detail: str = ""
    score: float = 1.0
    bbox: Optional[Tuple[float, float, float, float]] = None
    provisional: bool = False  # seen, but still waiting for future frames
    held_s: float = 0.0        # how long the candidate held before it was reported
    key: str = ""              # engine key (for retracting a vetoed event)
    evidence: dict = field(default_factory=dict)  # auditable why-it-fired facts


# --------------------------------------------------------------------------- #
# per-track state
# --------------------------------------------------------------------------- #
@dataclass
class _Pending:
    """A candidate event seen but not yet confirmed by future frames."""
    key: str
    first_seen: float
    last_seen: float
    event: Optional[Event]


@dataclass
class _Track:
    tid: int
    cls: str
    hist: Deque[Tuple[float, float, float]] = field(default_factory=deque)  # t, cx, cy
    boxes: Deque[Tuple[float, Tuple[float, float, float, float]]] = field(default_factory=deque)
    raw_speeds: Deque[Tuple[float, float]] = field(default_factory=deque)  # t, |v|
    dv_max: float = 0.0  # largest abrupt speed drop seen recently
    speed: float = 0.0            # EMA smoothed px/s
    peak: float = 0.0             # decaying recent peak (evidence of motion)
    stopped_in_jam: bool = False  # stop happened inside a jam -> not an incident
    last_seen: float = -1e9
    motion: Optional[float] = None  # mean abs frame difference inside the box
    motion_ema: float = 0.0
    stopped_since: Optional[float] = None

    def add(self, t: float, box: Tuple[float, float, float, float], cfg: EngineConfig,
            motion: Optional[float] = None) -> None:
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        if motion is not None:
            self.motion = motion
            self.motion_ema = (0.6 * self.motion_ema + 0.4 * motion
                               if self.motion_ema else motion)
        if self.hist:
            t0, x0, y0 = self.hist[-1]
            dt = t - t0
            if dt > 1e-3:
                v = math.hypot(cx - x0, cy - y0) / dt
                self.speed = cfg.speed_ema * v + (1.0 - cfg.speed_ema) * self.speed
                self.raw_speeds.append((t, v))
                self._update_dv(t)
        self.peak = max(self.peak * 0.985, self.speed)  # ~3s memory of fast motion
        self.hist.append((t, cx, cy))
        self.boxes.append((t, box))
        self.last_seen = t
        # trim
        while self.hist and t - self.hist[0][0] > cfg.history_s:
            self.hist.popleft()
        while self.raw_speeds and t - self.raw_speeds[0][0] > cfg.history_s:
            self.raw_speeds.popleft()
        while self.boxes and t - self.boxes[0][0] > cfg.history_s:
            self.boxes.popleft()
        if self.speed < cfg.stopped_min_speed:
            if self.stopped_since is None:
                self.stopped_since = t
        else:
            self.stopped_since = None
            self.stopped_in_jam = False

    @property
    def centroid(self) -> Tuple[float, float]:
        _, cx, cy = self.hist[-1]
        return cx, cy

    @property
    def box(self) -> Tuple[float, float, float, float]:
        return self.boxes[-1][1]

    def real_speed(self, cfg: EngineConfig) -> float:
        """Displacement speed with detector box-jitter removed.

        A parked car whose box wobbles a few pixels per frame reads as moving
        under plain centroid displacement. When we have the pixel-motion signal
        and the pixels inside the box are not changing, the displacement is
        noise and the vehicle counts as standing still.
        """
        if self.motion is None:
            return self.speed
        if self.motion_ema < cfg.motion_still_thresh and self.speed < cfg.jitter_speed_px_s:
            return 0.0
        return self.speed

    def _update_dv(self, t: float) -> None:
        """Track the largest speed drop across any 0.5s of recent samples.

        The scan keeps a running maximum, so it is the biggest drop from
        any earlier sample to any later one inside the span - exactly the
        "how fast did it fall" of a discontinuity. It decays slowly so the
        evidence is still there when the overlap condition is met a few
        frames (and one EMA time-constant) later.
        """
        drop = 0.0
        run_max = 0.0
        for tt, v in self.raw_speeds:
            if tt < t - 0.5:
                continue
            if v > run_max:
                run_max = v
            elif run_max - v > drop:
                drop = run_max - v
        self.dv_max = max(drop, self.dv_max * 0.99)

    def delta_v(self) -> float:
        """Abrupt deceleration (px/s) seen in the last few seconds."""
        return self.dv_max


# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #
class EventEngine:
    def __init__(self, cfg: Optional[EngineConfig] = None, frame_shape: Tuple[int, int] = (720, 1280)):
        self.cfg = cfg or EngineConfig()
        h, w = frame_shape
        self.frame_w, self.frame_h = w, h
        self.roi = (0, int(h * self.cfg.roi_top_frac), w, h)  # x1,y1,x2,y2
        self.tracks: Dict[int, _Track] = {}
        self.zones: Dict[Tuple[int, int], ZoneState] = {}
        for c in range(self.cfg.grid_cols):
            for r in range(self.cfg.grid_rows):
                self.zones[(c, r)] = ZoneState(col=c, row=r)
        # zone median-speed history for collapse detection: key -> deque[(t, speed)]
        self._zone_speed_hist: Dict[Tuple[int, int], Deque[Tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=int(self.cfg.history_s * 10))
        )
        # zone vehicle-count history: distinguishes collapse (same cars) from
        # queue growth (new cars arriving)
        self._zone_count_hist: Dict[Tuple[int, int], Deque[Tuple[float, int]]] = defaultdict(
            lambda: deque(maxlen=int(self.cfg.history_s * 10))
        )
        # per-event score + last fired time (for hysteresis / cooldown)
        self._scores: Dict[str, float] = defaultdict(float)
        self._last_fired: Dict[str, float] = defaultdict(lambda: -1e9)
        self._ped_acc: Dict[int, float] = {}  # pedestrian conflict evidence (s)
        # collision evidence (s): overlap alone is too noisy in dense traffic,
        # so it has to persist before we call it an accident
        self._collision_since: Optional[float] = None
        # candidates seen but not yet confirmed, awaiting future frames
        self._pending: Dict[str, "_Pending"] = {}
        # road mask from OpenPSG; None means "no constraint available"
        self.road_mask = None
        self.road_mask_h = self.road_mask_w = 0
        # slow-decaying reference to this clip's normal traffic speed (px/s)
        self._scene_speed_ref = 0.0
        self.active: Dict[str, Event] = {}  # currently fired events (by key)
        self.history: List[Event] = []      # fired (and since-cleared) events
        self.jam_origin_key: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------ util
    def _zone_of(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        x1, y1, x2, y2 = self.roi
        if not (x1 <= x <= x2 and y1 <= y <= y2):
            return None
        c = int((x - x1) / (x2 - x1) * self.cfg.grid_cols)
        r = int((y - y1) / (y2 - y1) * self.cfg.grid_rows)
        c = min(max(c, 0), self.cfg.grid_cols - 1)
        r = min(max(r, 0), self.cfg.grid_rows - 1)
        return (c, r)

    def zone_pixel_area(self) -> float:
        x1, y1, x2, y2 = self.roi
        return ((x2 - x1) / self.cfg.grid_cols) * ((y2 - y1) / self.cfg.grid_rows)

    def zone_rect(self, key: Tuple[int, int]) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.roi
        zw = (x2 - x1) / self.cfg.grid_cols
        zh = (y2 - y1) / self.cfg.grid_rows
        c, r = key
        return (int(x1 + c * zw), int(y1 + r * zh), int(x1 + (c + 1) * zw), int(y1 + (r + 1) * zh))

    def active_flow(self) -> bool:
        """Is the road generally flowing right now? (needed to judge 'stopped')"""
        speeds = [z.median_speed for z in self.zones.values() if z.vehicles >= 1]
        return _median(speeds, 0.0) > self.jam_threshold_px_s() * 1.5

    # ------------------------------------------------------- calibration / road
    def jam_threshold_px_s(self) -> float:
        """The speed below which a zone counts as congested, in px/s.

        "abs"    - the fixed historical threshold (default, unchanged behaviour)
        "metric" - a real-world speed, converted with the px-per-metre calibration
        "auto"   - a share of the traffic speed seen in this clip, floored at the
                   absolute value so it can never become absurdly small
        """
        cfg = self.cfg
        if cfg.jam_speed_mode == "metric" and cfg.px_per_meter:
            return max(1.0, cfg.jam_speed_mps * cfg.px_per_meter)
        if cfg.jam_speed_mode == "auto":
            return max(cfg.jam_speed_px_s, cfg.jam_speed_rel_frac * self._scene_speed_ref)
        return cfg.jam_speed_px_s

    def set_road_mask(self, mask) -> None:
        """Install a boolean road mask (OpenPSG stuff region), or None."""
        if mask is None:
            self.road_mask = None
            return
        self.road_mask = mask
        self.road_mask_h, self.road_mask_w = mask.shape[:2]

    def on_road(self, box: Tuple[float, float, float, float]) -> bool:
        """Is this box mostly on the road? True when no mask is available."""
        m = self.road_mask
        if m is None:
            return True
        x1, y1, x2, y2 = box
        h, w = m.shape[:2]
        sx, sy = w / float(max(1, self.frame_w)), h / float(max(1, self.frame_h))
        ix1, iy1 = int(max(0, x1 * sx)), int(max(0, y1 * sy))
        ix2, iy2 = int(min(w, max(ix1 + 1, x2 * sx))), int(min(h, max(iy1 + 1, y2 * sy)))
        sub = m[iy1:iy2, ix1:ix2]
        if sub.size == 0:
            return True
        return float(sub.mean()) >= 0.5

    # ---------------------------------------------------------------- update
    def update(self, t: float, detections) -> List[Event]:
        """detections: iterable of dicts {id, cls, bbox(xyxy), conf}."""
        cfg = self.cfg

        # 1) ingest track states
        seen_ids = set()
        for d in detections:
            tid = int(d["id"])
            cls = str(d["cls"]).lower()
            box = tuple(map(float, d["bbox"]))
            tr = self.tracks.get(tid)
            if tr is None:
                tr = _Track(tid=tid, cls=cls)
                self.tracks[tid] = tr
            tr.cls = cls
            tr.add(t, box, cfg, motion=d.get("motion"))
            seen_ids.add(tid)
        # prune stale tracks
        for tid in [k for k, v in self.tracks.items() if t - v.last_seen > cfg.history_s]:
            del self.tracks[tid]

        # 2) zone stats
        for z in self.zones.values():
            z.vehicles = 0
            z.pedestrians = 0
        zone_speeds: Dict[Tuple[int, int], List[float]] = defaultdict(list)
        moving_speeds: List[float] = []
        for tr in self.tracks.values():
            if t - tr.last_seen > 0.5:
                continue
            cx, cy = tr.centroid
            key = self._zone_of(cx, cy)
            if key is None:
                continue
            if tr.cls in VEHICLE_CLASSES:
                # "standing on the road" has to mean the road: with a PSG road
                # mask available, cars in a car park or on a pavement no longer
                # count towards congestion.
                if not self.on_road(tr.box):
                    continue
                spd = tr.real_speed(cfg)
                self.zones[key].vehicles += 1
                zone_speeds[key].append(spd)
                if spd > 0:
                    moving_speeds.append(spd)
            elif tr.cls == "person":
                self.zones[key].pedestrians += 1

        # what "normal traffic speed" looks like in this clip, so the jam
        # threshold can be relative to it rather than an absolute px/s
        if moving_speeds:
            obs = _median(moving_speeds, 0.0)
            self._scene_speed_ref = max(self._scene_speed_ref * 0.995, obs, 1.0)
        area_k = max(1e-6, self.zone_pixel_area() / 1000.0)
        for key, z in self.zones.items():
            z.median_speed = _median(zone_speeds.get(key, []), 0.0)
            # only record speed while the zone actually has cars — an empty
            # zone's default 0 must not look like a speed collapse later
            if z.vehicles >= 1:
                self._zone_speed_hist[key].append((t, z.median_speed))
            self._zone_count_hist[key].append((t, z.vehicles))

        fired: List[Event] = []
        self._check_jam(t, fired)
        self._check_accident(t, fired, zone_speeds)
        self._check_stopped(t, fired, area_k)
        self._check_pedestrian(t, fired)
        return fired

    # ----------------------------------------------------------------- jams
    def _check_jam(self, t: float, fired: List[Event]) -> None:
        cfg = self.cfg
        jammed_now = {}
        thresh = self.jam_threshold_px_s()
        for key, z in self.zones.items():
            density = z.vehicles / area_k if (area_k := max(1e-6, self.zone_pixel_area() / 1000.0)) else 0.0
            bad = (
                z.vehicles >= cfg.jam_min_vehicles
                and (z.median_speed < thresh or density >= cfg.jam_min_density * 2)
            )
            if bad:
                if z.jam_onset_t is None:
                    z.jam_onset_t = t
                # persistence handled below
                held = t - z.jam_onset_t >= cfg.jam_persist_s
                z.jammed = held
                if held:
                    jammed_now[key] = z
            else:
                z.jam_onset_t = None
                z.jammed = False
                z.jam_score = 0.0

        self._emit(
            fired,
            key="JAM",
            t=t,
            score=1.0 if jammed_now else 0.0,
            event=Event(
                type="JAM",
                t=t,
                zone=None,
                detail=f"{len(jammed_now)} congested zone(s)"
                + (f", slowest {_median([z.median_speed for z in jammed_now.values()]):.0f} px/s" if jammed_now else ""),
                evidence=(
                    {"slowest_px_s": round(_median([z.median_speed for z in jammed_now.values()]), 1)}
                    if jammed_now else {}
                ),
            ),
            extra_zones=set(jammed_now),
        )

        # ---- jam origin: earliest onset among currently-jammed zones -----
        origin = None
        if jammed_now:
            def onset(z):
                return z.jam_onset_t if z.jam_onset_t is not None else t
            origin = min(jammed_now.values(), key=onset)
            # require it to be clearly earlier than the rest
            onsets = sorted(onset(z) for z in jammed_now.values())
            if len(onsets) >= 2 and onsets[1] - onset(origin) < cfg.jam_origin_margin_s:
                # tie: pick the one with lowest median speed (most saturated)
                origin = min(jammed_now.values(), key=lambda z: z.median_speed)
        self.jam_origin_key = origin.key if origin else None
        self._emit(
            fired,
            key="JAM_ORIGIN",
            t=t,
            score=1.0 if origin else 0.0,
            event=Event(
                type="JAM_ORIGIN",
                t=t,
                zone=origin.key if origin else None,
                detail=(f"queue started here at t={origin.jam_onset_t:.1f}s"
                        if origin and origin.jam_onset_t is not None else "")
                if origin
                else "",
            ),
        )

        # ---- jam front: flowing zone adjacent to a jammed zone ------------
        front = None
        if jammed_now:
            for key, z in jammed_now.items():
                c, r = key
                for nc, nr in ((c - 1, r), (c + 1, r), (c, r - 1), (c, r + 1)):
                    nb = self.zones.get((nc, nr))
                    if nb is None or nb.jammed:
                        continue
                    if nb.vehicles >= 1 and nb.median_speed >= cfg.jam_front_speed_px_s:
                        front = (nc, nr)
                        break
                if front:
                    break
        self._emit(
            fired,
            key="JAM_FRONT",
            t=t,
            score=1.0 if front else 0.0,
            event=Event(type="JAM_FRONT", t=t, zone=front, detail="queue head / flow resumes"),
        )

    # ------------------------------------------------------------- accidents
    def _check_accident(self, t: float, fired: List[Event], zone_speeds) -> None:
        cfg = self.cfg
        # (a) collision-like overlap of two recently-fast vehicles
        vehicles = [tr for tr in self.tracks.values() if tr.cls in MOVING_CLASSES and t - tr.last_seen <= 0.5]
        collision: Optional[Event] = None
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                a, b = vehicles[i], vehicles[j]
                if math.hypot(a.centroid[0] - b.centroid[0], a.centroid[1] - b.centroid[1]) > 250:
                    continue
                iou = iou_xyxy(a.box, b.box)
                if iou >= cfg.collision_iou and max(a.speed, b.speed) < cfg.jam_speed_px_s \
                        and max(a.peak, b.peak) >= cfg.collision_min_speed * 0.5:
                    dv = max(a.delta_v(), b.delta_v())
                    if dv < cfg.collision_dv_px_s:
                        # overlap without a velocity jump: a merge or a
                        # queue bumper, not an impact
                        continue
                    # boxes overlap, both stopped now, they were moving fast
                    # recently AND their speed collapsed abruptly -> impact
                    collision = Event(
                        type="ACCIDENT",
                        t=t,
                        track_ids=(a.tid, b.tid),
                        detail=f"collision-like overlap IoU={iou:.2f}, dv={dv:.0f}px/s",
                        bbox=_union_box(a.box, b.box),
                        evidence={
                            "iou": round(iou, 2),
                            "dv_px_s": round(dv, 1),
                            "speeds_px_s": (round(a.speed, 1), round(b.speed, 1)),
                            "peaks_px_s": (round(a.peak, 1), round(b.peak, 1)),
                        },
                    )
                    break
            if collision:
                break

        # (b) zone speed collapse: median fell >= collapse_drop_frac within window
        #     AND the vehicle count did not grow (queue growth is not a crash)
        collapse: Optional[Event] = None
        for key, hist in self._zone_speed_hist.items():
            vals = [(tt, v) for tt, v in hist if t - tt <= 4.0]
            if len(vals) < 6:
                continue
            recent = _median([v for tt, v in vals if t - tt <= 1.0], 0.0)
            before_vals = [v for tt, v in vals if 1.5 <= t - tt <= 4.0]
            if not before_vals:
                continue
            before = _median(before_vals, 0.0)
            if before >= cfg.collapse_min_before and recent <= before * (1 - cfg.collapse_drop_frac):
                chist = self._zone_count_hist.get(key, [])
                cnt_recent = _median([c for tt, c in chist if t - tt <= 1.0], 0)
                cnt_before = _median([c for tt, c in chist if 1.5 <= t - tt <= 4.0], 0)
                if cnt_recent > cnt_before + 1:
                    continue  # cars are piling up -> queue, not collapse
                z = self.zones[key]
                if z.vehicles >= 2:
                    collapse = Event(
                        type="ACCIDENT",
                        t=t,
                        zone=key,
                        detail=f"speed collapsed {before:.0f}->{recent:.0f} px/s",
                        evidence={
                            "before_px_s": round(before, 1),
                            "recent_px_s": round(recent, 1),
                            "drop_frac": round(cfg.collapse_drop_frac, 2),
                        },
                    )
                    break

        # the overlap must persist; one frame of jitter is just occlusion
        if collision is not None:
            if self._collision_since is None:
                self._collision_since = t
            held = t - self._collision_since
            collision.detail += f", held {held:.1f}s"
            if held < cfg.collision_persist_s:
                collision = None  # not yet convincing enough to fire
            collision_score = 1.0 if collision is not None else 0.9 * held / max(1e-3, cfg.collision_persist_s)
        else:
            self._collision_since = None
            collision_score = 0.0

        ev = collision or collapse
        self._emit(
            fired,
            key="ACCIDENT",
            t=t,
            score=collision_score if collision_score > 0 else (1.0 if collapse else 0.0),
            event=ev or Event(type="ACCIDENT", t=t),
            cool_down_override=10.0,
        )

    # ---------------------------------------------------- stopped vehicles
    def _check_stopped(self, t: float, fired: List[Event], area_k: float) -> None:
        cfg = self.cfg
        if not self.active_flow():
            return  # whole road is jammed; standing still is expected
        for tr in self.tracks.values():
            if tr.cls not in VEHICLE_CLASSES or t - tr.last_seen > 0.5:
                continue
            if tr.stopped_since is None:
                continue
            cx, cy = tr.centroid
            key = self._zone_of(cx, cy)
            if key is not None and self.zones[key].jammed:
                tr.stopped_in_jam = True  # remember: this stop is congestion
            if tr.stopped_in_jam:
                continue
            dur = t - tr.stopped_since
            if dur < cfg.stopped_min_s:
                continue
            # nose-to-tail with another stopped vehicle = queueing, not incident
            near_stop = any(
                o.tid != tr.tid
                and o.cls in VEHICLE_CLASSES
                and t - o.last_seen <= 0.5
                and o.speed < cfg.stopped_min_speed
                and math.hypot(o.centroid[0] - cx, o.centroid[1] - cy) < 170
                for o in self.tracks.values()
            )
            if near_stop:
                continue
            self._emit(
                fired,
                key=f"STOPPED:{tr.tid}",
                t=t,
                score=1.0,
                event=Event(
                    type="STOPPED",
                    t=t,
                    zone=key,
                    track_ids=(tr.tid,),
                    detail=f"{tr.cls}#{tr.tid} stationary {dur:.1f}s in active traffic",
                    bbox=tr.box,
                    evidence={"stopped_s": round(dur, 1), "speed_px_s": round(tr.speed, 1)},
                ),
                cool_down_override=12.0,
            )

    # ------------------------------------------------------ pedestrian conflict
    def _check_pedestrian(self, t: float, fired: List[Event]) -> None:
        cfg = self.cfg
        # accumulate evidence over time so a single frame never fires
        cond_active = set()
        for tr in self.tracks.values():
            if tr.cls != "person" or t - tr.last_seen > 0.5:
                continue
            cx, cy = tr.centroid
            key = self._zone_of(cx, cy)
            if key is None:
                continue
            z = self.zones[key]
            if z.vehicles < cfg.ped_conflict_min_vehicles:
                continue
            cond_active.add(tr.tid)
            acc = self._ped_acc.get(tr.tid, 0.0) + 0.1  # one frame ≈ 0.1s
            self._ped_acc[tr.tid] = acc
            if acc < cfg.ped_conflict_persist_s:
                continue
            self._emit(
                fired,
                key=f"PED:{tr.tid}",
                t=t,
                score=1.0,
                event=Event(
                    type="PED_CONFLICT",
                    t=t,
                    zone=key,
                    track_ids=(tr.tid,),
                    detail=f"person#{tr.tid} in roadway with {z.vehicles} vehicles",
                    bbox=tr.box,
                ),
                cool_down_override=8.0,
            )
        # decay accumulators for pedestrians no longer in conflict
        for tid in list(self._ped_acc):
            if tid not in cond_active:
                self._ped_acc[tid] = max(0.0, self._ped_acc[tid] - 0.3)
                if self._ped_acc[tid] == 0.0:
                    del self._ped_acc[tid]

    # ------------------------------------------------------------------ emit
    def _emit(
        self,
        fired: List[Event],
        *,
        key: str,
        t: float,
        score: float,
        event: Optional[Event],
        extra_zones=None,
        cool_down_override: Optional[float] = None,
    ) -> None:
        """Hysteresis: fire when score reaches on-threshold, clear when it
        drops below off-threshold.  Cooldown suppresses rapid repeats.

        A new candidate is not reported straight away: it waits `confirm_s` for
        this key, and if the score falls apart before then it is dropped without
        ever appearing in the log. That is the "look at the next frames" half of
        the decision, which no amount of past-frame evidence can replace.
        """
        cfg = self.cfg
        cur = self._scores.get(key, 0.0)
        cur = score if score > 0 else max(0.0, cur - 1.0)
        cur = min(cur, cfg.event_max)
        self._scores[key] = cur

        need = getattr(cfg, CONFIRM_ATTR.get(key, ""), 0.0) or 0.0
        was = key in self.active
        if not was and cur >= cfg.event_on:
            cool = cool_down_override if cool_down_override is not None else cfg.cool_down_s
            pend = self._pending.get(key)
            if pend is None:
                pend = _Pending(key=key, first_seen=t, last_seen=t, event=event)
                self._pending[key] = pend
            else:
                pend.last_seen = t
                if event is not None:
                    pend.event = event  # keep the freshest bbox / detail
            if (t - pend.first_seen >= need and t - self._last_fired[key] >= cool
                    and pend.event is not None):
                ev = pend.event
                ev.key = key
                ev.provisional = False
                ev.held_s = round(t - pend.first_seen, 3)
                self.active[key] = ev
                self._last_fired[key] = t
                if extra_zones is not None:
                    ev.detail = ev.detail + f" zones={sorted(extra_zones)}"
                self.history.append(ev)
                fired.append(ev)
                del self._pending[key]
        else:
            if cur < cfg.event_off:
                # retracted before it was ever confirmed -> never reported
                self._pending.pop(key, None)
            if was and cur < cfg.event_off:
                del self.active[key]

    def retract(self, key: str) -> None:
        """Withdraw a fired event (e.g. a VLM adjudicator veto).

        The score resets so the rule can fire again after its normal
        cooldown, giving the vetoed decision a second chance on the
        next frames.
        """
        self.active.pop(key, None)
        self._scores[key] = 0.0

    # ------------------------------------------------------ confirmation api
    def provisional_events(self) -> List[Event]:
        """Candidates seen but not yet confirmed - draw them, but do not log."""
        return [p.event for p in self._pending.values() if p.event is not None]

    def finalize(self, t: float) -> List[Event]:
        """Flush still-valid candidates at end of stream.

        Without this, anything that happens in the last `confirm_s` seconds of a
        clip would be silently dropped, because confirmation needs frames that
        never arrive.
        """
        out: List[Event] = []
        for key, pend in list(self._pending.items()):
            if t - pend.last_seen > 1e-6 or pend.event is None:
                continue
            ev = pend.event
            ev.key = key
            ev.provisional = False
            ev.held_s = round(t - pend.first_seen, 3)
            self.active[key] = ev
            self.history.append(ev)
            out.append(ev)
            del self._pending[key]
        return out

    # ----------------------------------------------------------------- api
    @property
    def active_events(self) -> List[Event]:
        return list(self.active.values())

    def speed_kmh(self, px_per_s: float, px_per_meter: Optional[float]) -> Optional[float]:
        if not px_per_meter:
            return None
        return px_per_s / px_per_meter * 3.6


def _union_box(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
