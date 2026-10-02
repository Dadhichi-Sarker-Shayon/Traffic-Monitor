"""Unit tests for traffic_watch.events — no GPU / MM stack needed."""

import math

import pytest

from traffic_watch.events import EngineConfig, EventEngine, iou_xyxy


W, H = 1280, 720


def make_engine(**kw):
    cfg = EngineConfig(**kw)
    return EventEngine(cfg, frame_shape=(H, W))


def det(tid, cls, x, y, w=60, h=40):
    return {"id": tid, "cls": cls, "bbox": (x, y, x + w, y + h), "conf": 0.9}


def step(engine, t, dets):
    return engine.update(t, dets)


def fired_types(engine):
    return {e.type for e in engine.active_events}


# --------------------------------------------------------------------------- #
def test_iou():
    assert iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou_xyxy((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3, abs=0.01)


def test_no_jam_when_flow_is_healthy():
    eng = make_engine()
    # 3 vehicles moving fast across the frame for 5s
    t = 0.0
    for i in range(50):
        t = i * 0.1
        dets = [det(j, "car", 200 + i * 60 + j * 150, 500) for j in range(3)]
        step(eng, t, dets)
    assert "JAM" not in fired_types(eng)


def test_jam_fires_and_origin_is_earliest_zone():
    eng = make_engine(grid_cols=4, grid_rows=2, roi_top_frac=0.45)
    # Vehicles pile up, barely moving, starting in the LEFT zone (col 0),
    # then a second zone (col 1) jams 5s later.
    t = 0.0
    fps_dt = 0.2
    for i in range(60):  # 12 seconds
        t = i * fps_dt
        dets = [
            det(1, "car", 100, 600),        # zone col0 / row2
            det(2, "car", 160, 640),
            det(3, "car", 120, 680),
        ]
        if t >= 5.0:                        # second zone jams later
            dets += [
                det(4, "car", 480, 600),    # zone col1 / row2
                det(5, "car", 540, 640),
                det(6, "car", 500, 680),
            ]
        step(eng, t, dets)
    types = fired_types(eng)
    assert "JAM" in types
    assert "JAM_ORIGIN" in types
    # origin must be the first zone (col 0), not the later one
    assert eng.jam_origin_key is not None
    assert eng.jam_origin_key[0] == 0


def test_stopped_vehicle_in_active_flow():
    eng = make_engine()
    t = 0.0
    for i in range(100):  # 10s
        t = i * 0.1
        dets = [
            det(1, "car", 640, 600),                                    # broken down, never moves
            det(2, "car", 100 + (i * 80) % 1150, 520),                  # flows past (wraps)
            det(3, "car", 60 + (i * 75) % 1150, 480),
            det(4, "car", 140 + (i * 85) % 1150, 560),
        ]
        step(eng, t, dets)
    stopped = [e for e in eng.active_events if e.type == "STOPPED"]
    assert stopped, "expected a STOPPED event for the non-moving car"
    assert 1 in stopped[0].track_ids


def test_pedestrian_conflict_in_roadway():
    eng = make_engine()
    t = 0.0
    for i in range(40):
        t = i * 0.1
        dets = [
            det(1, "person", 600, 580, w=30, h=60),       # in the roadway
            det(2, "car", 450 + (i * 20) % 150, 600),      # slowly moving, same zone
            det(3, "car", 470 + (i * 25) % 130, 620),
        ]
        step(eng, t, dets)
    assert "PED_CONFLICT" in fired_types(eng)


def test_collision_like_overlap_fires_accident():
    # the crash is now confirmed against future frames (confirm_accident_s=1.5
    # on top of collision_persist_s=0.6), so the wreck has to be held for ~2.1s
    eng = make_engine()
    t = 0.0
    for i in range(45):
        t = i * 0.1
        if i < 10:
            # two cars approaching fast (moving 90px / 0.1s = 900 px/s)
            dets = [det(1, "car", 100 + i * 90, 550), det(2, "car", 300 + i * 90, 555)]
        else:
            # they crashed: overlapping and both stopped
            dets = [det(1, "car", 990, 550), det(2, "car", 1010, 552)]
        step(eng, t, dets)
    assert "ACCIDENT" in fired_types(eng)


def test_transient_overlap_does_not_fire_accident():
    """A single frame of box overlap (ID swap / occlusion in dense traffic)
    must not raise an accident: the condition has to persist."""
    eng = make_engine(collision_persist_s=0.6)
    t = 0.0
    for i in range(30):
        t = i * 0.1
        if i < 10:
            dets = [det(1, "car", 100 + i * 90, 550), det(2, "car", 300 + i * 90, 555)]
        elif i < 12:  # two frames of jitter-level overlap, then they separate
            dets = [det(1, "car", 990, 550), det(2, "car", 1010, 552)]
        else:
            dets = [det(1, "car", 990 + i, 550), det(2, "car", 1200, 552)]
        step(eng, t, dets)
    assert "ACCIDENT" not in fired_types(eng)


def test_hysteresis_clears_events():
    eng = make_engine(jam_persist_s=0.5)
    t = 0.0
    # create a jam
    for i in range(20):
        t = i * 0.2
        dets = [det(j, "car", 100 + j * 60, 600) for j in range(1, 5)]
        step(eng, t, dets)
    assert "JAM" in fired_types(eng)
    # everyone leaves
    for i in range(60):
        t = 20 + i * 0.2
        step(eng, t, [])
    assert "JAM" not in fired_types(eng)


def test_engine_survives_empty_frames():
    eng = make_engine()
    for i in range(10):
        eng.update(i * 0.1, [])
    assert eng.active_events == []


# --------------------------------------------------------------------------- #
# retrospective confirmation: a candidate has to survive future frames
# --------------------------------------------------------------------------- #
def test_candidate_is_not_reported_before_the_confirmation_window():
    eng = make_engine(jam_persist_s=0.5, confirm_jam_s=1.0)
    t = 0.0
    for i in range(20):
        t = i * 0.2
        step(eng, t, [det(j, "car", 100 + j * 60, 600) for j in range(1, 5)])
        if t < 1.0:
            assert "JAM" not in fired_types(eng), "fired before the window elapsed"
    assert "JAM" in fired_types(eng)


def test_candidate_that_dissolves_is_retracted_silently():
    """The whole point: seeing the next frames can cancel a decision.

    Congestion lasts 2.0s: long enough to satisfy jam_persist_s=0.5 and become
    a candidate, far short of confirm_jam_s=3.0, so the queue dissolving has to
    retract it before it is ever reported.
    """
    eng = make_engine(jam_persist_s=0.5, confirm_jam_s=3.0)
    t = 0.0
    for i in range(10):  # 0.0 - 1.8s of congestion -> candidate only
        t = i * 0.2
        step(eng, t, [det(j, "car", 100 + j * 60, 600) for j in range(1, 5)])
    assert len(eng.history) == 0, "reported before it was confirmed"
    assert eng.provisional_events(), "should still be a provisional candidate"
    for i in range(30):  # road clears before the window elapses
        t = 4 + i * 0.2
        step(eng, t, [])
    assert len(eng.history) == 0, "retracted candidate must never be reported"
    assert not eng.provisional_events()


def test_finalize_flushes_candidates_at_end_of_stream():
    eng = make_engine(jam_persist_s=0.5, confirm_jam_s=5.0)
    t = 0.0
    for i in range(20):
        t = i * 0.2
        step(eng, t, [det(j, "car", 100 + j * 60, 600) for j in range(1, 5)])
    assert len(eng.history) == 0
    flushed = eng.finalize(t)
    assert "JAM" in [e.type for e in flushed], flushed
    assert "JAM" in fired_types(eng)


def test_confirm_zero_restores_immediate_firing():
    eng = make_engine(jam_persist_s=0.5, confirm_jam_s=0.0)
    t = 0.0
    for i in range(20):
        t = i * 0.2
        step(eng, t, [det(j, "car", 100 + j * 60, 600) for j in range(1, 5)])
    assert "JAM" in fired_types(eng)


# --------------------------------------------------------------------------- #
# motion: displacement is not movement
# --------------------------------------------------------------------------- #
def det_moving(tid, cls, x, y, motion, w=60, h=40):
    d = det(tid, cls, x, y, w, h)
    d["motion"] = motion
    return d


def test_box_jitter_is_not_treated_as_motion():
    """A stationary car whose box wanders must still count as standing still."""
    eng = make_engine(confirm_jam_s=0.0)
    t = 0.0
    for i in range(40):
        t = i * 0.2
        dets = [det_moving(j, "car", 100 + j * 60 + (7 if i % 2 else 0), 600, motion=0.002)
                for j in range(1, 5)]
        step(eng, t, dets)
    assert "JAM" in fired_types(eng), "jittering boxes were mistaken for a moving queue"
    assert eng.zones[(0, 2)].median_speed == 0.0


def test_real_motion_defeats_the_jitter_filter():
    """Same jitter, but the pixels are changing: that really is movement."""
    eng = make_engine(confirm_jam_s=0.0)
    t = 0.0
    for i in range(40):
        t = i * 0.2
        dets = [det_moving(j, "car", 100 + j * 60 + (7 if i % 2 else 0), 600, motion=0.20)
                for j in range(1, 5)]
        step(eng, t, dets)
    assert "JAM" not in fired_types(eng)


# --------------------------------------------------------------------------- #
# the road mask: "on the road" has to mean the road
# --------------------------------------------------------------------------- #
def test_cars_off_the_road_do_not_count_towards_a_jam():
    import numpy as np

    eng = make_engine(confirm_jam_s=0.0)
    mask = np.zeros((H, W), dtype=bool)
    mask[600:700, :] = True  # only a strip at the bottom is road
    eng.set_road_mask(mask)
    t = 0.0
    for i in range(40):
        t = i * 0.2
        step(eng, t, [det(j, "car", 100 + j * 60, 300) for j in range(1, 5)])  # y=300, off road
    assert "JAM" not in fired_types(eng), "counted cars that are not on the road"
    for i in range(40):
        t = 10 + i * 0.2
        step(eng, t, [det(j, "car", 100 + j * 60, 620) for j in range(1, 5)])  # y=620, on road
    assert "JAM" in fired_types(eng)


def test_no_road_mask_means_no_constraint():
    eng = make_engine(confirm_jam_s=0.0)
    t = 0.0
    for i in range(40):
        t = i * 0.2
        step(eng, t, [det(j, "car", 100 + j * 60, 600) for j in range(1, 5)])
    assert "JAM" in fired_types(eng)


# --------------------------------------------------------------------------- #
# scale-free jam thresholds
# --------------------------------------------------------------------------- #
def test_auto_threshold_follows_the_clip_speed():
    """A 300 px/s road must not be judged by the same 18 px/s as a 60 px/s one."""
    eng = make_engine(confirm_jam_s=0.0, jam_speed_mode="auto")
    assert eng.jam_threshold_px_s() == eng.cfg.jam_speed_px_s  # floor before any evidence
    t = 0.0
    for i in range(40):  # fast flowing traffic
        t = i * 0.2
        step(eng, t, [det(j, "car", 100 * i + j * 60, 600) for j in range(1, 5)])
    fast = eng.jam_threshold_px_s()
    assert fast > eng.cfg.jam_speed_px_s, "threshold did not scale with the scene"
    assert "JAM" not in fired_types(eng)


def test_metric_threshold_uses_the_calibration():
    eng = make_engine(jam_speed_mode="metric", px_per_meter=20.0, jam_speed_mps=2.0)
    assert eng.jam_threshold_px_s() == pytest.approx(40.0)
