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
    # the crash is confirmed against future frames (confirm_accident_s on top of
    # collision_persist_s) and needs vehicles that were established before contact
    eng = make_engine()
    t = 0.0
    for i in range(60):
        t = i * 0.1
        if i < 20:
            # two cars approaching fast (45px / 0.1s = 450 px/s)
            dets = [det(1, "car", 100 + i * 45, 550), det(2, "car", 150 + i * 45, 555)]
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


# --------------------------------------------------------------------------- #
# velocity discontinuity: an impact is abrupt, a merge is not
# --------------------------------------------------------------------------- #
def test_delta_v_spikes_on_abrupt_stop():
    eng = make_engine()
    t = 0.0
    for i in range(30):  # 90 px/s, then a dead stop
        t = i * 0.1
        x = 100 + i * 9 if i < 15 else 100 + 15 * 9
        step(eng, t, [det(1, "car", x, 550)])
    assert eng.tracks[1].delta_v() >= 60, eng.tracks[1].dv_max


def test_delta_v_stays_low_for_gradual_braking():
    eng = make_engine()
    t = 0.0
    for i in range(60):  # linear brake 90 -> 0 px/s over 6s
        t = i * 0.1
        x = 100 + 90 * t - 7.5 * t * t
        step(eng, t, [det(1, "car", x, 550)])
    assert eng.tracks[1].delta_v() < 25, eng.tracks[1].dv_max


def test_gentle_merge_with_overlap_does_not_fire_accident():
    """Overlap + both stopped + was fast is not enough without a
    velocity jump: braking gently into a slight overlap is a merge."""
    eng = make_engine()
    t = 0.0
    for i in range(110):  # 11s
        t = i * 0.1
        if i < 20:
            # car 1 cruises at 90 px/s; car 2 parked ahead
            dets = [det(1, "car", 100 + i * 9, 550), det(2, "car", 584, 552)]
        else:
            # gentle 6s brake (90 -> 0 px/s) into a slight overlap
            k = (i - 20) * 0.1
            x = 280 + 90 * k - 7.5 * k * k
            dets = [det(1, "car", x, 550), det(2, "car", 584, 552)]
        step(eng, t, dets)
    assert "ACCIDENT" not in fired_types(eng)


def test_abrupt_stop_with_overlap_fires_accident():
    eng = make_engine()
    t = 0.0
    for i in range(75):
        t = i * 0.1
        if i < 40:
            # car 1 cruises at 90 px/s toward a parked car
            dets = [det(1, "car", 100 + i * 9, 550), det(2, "car", 500, 552)]
        else:
            # brakes dead, overlapping the parked car (IoU ~0.31)
            dets = [det(1, "car", 470, 550), det(2, "car", 500, 552)]
        step(eng, t, dets)
    fired = [e for e in eng.active_events if e.type == "ACCIDENT"]
    assert fired, "expected ACCIDENT for an abrupt stop with overlap"
    assert fired[0].evidence.get("dv_px_s", 0) >= 25
    assert 1 in fired[0].track_ids and 2 in fired[0].track_ids


# --------------------------------------------------------------------------- #
# scene-level jam (net displacement in body-widths/s)
# --------------------------------------------------------------------------- #
def _queue(speed_px_s, n=10, secs=7.0, fps=25):
    """n cars in two rows drifting at speed_px_s; returns the engine."""
    eng = make_engine()
    for f in range(int(secs * fps)):
        t = f / fps
        dets = [det(i + 1, "car", 100 + 110 * (i % 5) + speed_px_s * t, 480 + 80 * (i // 5))
                for i in range(n)]
        step(eng, t, dets)
    return eng


def test_net_speed_ignores_box_jitter():
    eng = make_engine()
    for f in range(60):
        t = f / 30
        jitter = 6 if f % 2 else -6          # +-6px every frame = 360 px/s "speed"
        step(eng, t, [det(1, "car", 300 + jitter, 500)])
    tr = eng.tracks[1]
    assert tr.speed > 100                     # the per-frame estimate is fooled
    assert tr.net_speed() < 15                # displacement over 1s is not


def test_crawling_scene_is_a_jam():
    eng = _queue(speed_px_s=8)                # ~0.13 widths/s
    assert eng._scene_crawling
    assert "JAM" in {e.type for e in eng.history}


def test_flowing_scene_is_not_a_jam():
    eng = _queue(speed_px_s=120)              # ~2 widths/s
    assert not eng._scene_crawling
    assert "JAM" not in {e.type for e in eng.history}


def test_too_few_vehicles_is_not_a_scene_jam():
    eng = _queue(speed_px_s=8, n=4)
    assert not eng._scene_crawling


def test_no_overlap_accident_inside_a_crawling_queue():
    """Bumper-to-bumper contact is normal while the whole road crawls."""
    eng = make_engine()
    for f in range(200):
        t = f / 25
        dets = [det(i + 1, "car", 100 + 110 * (i % 5) + 8 * t, 480 + 80 * (i // 5)) for i in range(10)]
        dets.append(det(50, "car", 150 + 8 * t, 482))   # overlaps car 1
        step(eng, t, dets)
    assert "ACCIDENT" not in {e.type for e in eng.history}


# --------------------------------------------------------------------------- #
# pedestrian conflict
# --------------------------------------------------------------------------- #
def _ped_scene(ped_kw=None, car_speed=150, car_x0=0, car_cls="car", secs=6.0, fps=25):
    """A person stands on the road; a car 90px wide passes (or sits beside) them."""
    eng = make_engine()
    ped_kw = ped_kw or {}
    for f in range(int(secs * fps)):
        t = f / fps
        dets = [
            det(1, "person", ped_kw.get("x", 600), ped_kw.get("y", 540),
                w=ped_kw.get("w", 30), h=ped_kw.get("h", 70)),
            det(2, car_cls, car_x0 + car_speed * t, 560, w=90, h=50),
        ]
        step(eng, t, dets)
    eng.finalize(secs)
    return eng


def _ped_events(eng):
    return [e for e in eng.history if e.type == "PED_CONFLICT"]


def test_pedestrian_next_to_a_moving_car_is_a_conflict():
    eng = _ped_scene()
    ev = _ped_events(eng)
    assert ev and ev[0].track_ids == (1, 2)
    assert ev[0].evidence["vehicle_id"] == 2


def test_pedestrian_next_to_a_stationary_car_is_not():
    assert not _ped_events(_ped_scene(car_speed=0, car_x0=520))


def test_rider_on_a_motorcycle_is_not_a_pedestrian():
    # person box sits on the motorbike box and moves with it
    eng = make_engine()
    for f in range(75):
        t = f / 25
        x = 380 + 300 * t
        step(eng, t, [det(1, "person", x + 30, 520, w=30, h=70),
                      det(2, "motorcycle", x, 560, w=90, h=50)])
    assert not _ped_events(eng)


def test_squarish_person_box_is_not_a_pedestrian():
    # a seated torso (aspect < 1.6) is a rider whose bike was not detected
    assert not _ped_events(_ped_scene(ped_kw={"w": 60, "h": 70}))


def test_a_pedestrian_cluster_raises_one_alert():
    eng = make_engine()
    for f in range(150):
        t = f / 25
        dets = [det(2, "car", 150 * t, 560, w=90, h=50)]
        dets += [det(10 + k, "person", 560 + 25 * k, 540, w=25, h=65) for k in range(4)]
        step(eng, t, dets)
    eng.finalize(6.0)
    assert len(_ped_events(eng)) == 1


def test_young_person_track_is_ignored():
    eng = make_engine()
    for f in range(8):                         # < ped_min_track_s
        t = f / 25
        step(eng, t, [det(1, "person", 600, 540, w=30, h=70),
                      det(2, "car", 380 + 300 * t, 560, w=90, h=50)])
    assert not _ped_events(eng)


# --------------------------------------------------------------------------- #
# pile-up sanity gates
# --------------------------------------------------------------------------- #
def _pileup(age_before_s, peak_px_s):
    eng = make_engine()
    fps, t = 25, 0.0
    n_before = int(age_before_s * fps)
    for f in range(n_before + 150):
        t = f / fps
        if f < n_before:
            x = 100 + peak_px_s * t
            dets = [det(1, "car", x, 540, w=80, h=50), det(2, "car", x + 100, 545, w=80, h=50)]
        else:
            x = 100 + peak_px_s * (n_before / fps)
            dets = [det(1, "car", x, 540, w=80, h=50), det(2, "car", x + 70, 545, w=80, h=50)]
        step(eng, t, dets)
    eng.finalize(t)
    return [e for e in eng.history if e.type == "ACCIDENT"]


def test_established_vehicles_stopping_together_is_reported():
    assert _pileup(age_before_s=4.0, peak_px_s=200)


def test_young_tracks_cannot_raise_a_pileup():
    assert not _pileup(age_before_s=1.0, peak_px_s=200)


def test_impossible_speeds_cannot_raise_a_pileup():
    # 80px cars at 4000 px/s = 50 widths/s: an id jump, not a vehicle
    assert not _pileup(age_before_s=4.0, peak_px_s=4000)
