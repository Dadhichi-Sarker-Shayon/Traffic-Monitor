"""Integration test: the live pipeline path on the synthetic demo clip.

The unit tests in test_events.py feed synthetic detection sequences
straight into the engine. This test drives the same code a live
monitoring session runs - real video capture (frame indices and media
timestamps), detections in the tracker's output format - and asserts
the events the clip was built to produce.

It needs no GPU and no models: the demo ships with demo_gt.jsonl,
which bypasses YOLO, and OpenPSG is off, so the whole path runs on
numpy + OpenCV alone. Rendering and video writing are presentation,
so they are left out; the decision path is what is under test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from traffic_watch.capture import open_source
from traffic_watch.events import EngineConfig, EventEngine

ROOT = Path(__file__).resolve().parent.parent
CLIP = ROOT / "sample_media" / "clips" / "demo_traffic.mp4"
GT = ROOT / "sample_media" / "clips" / "demo_gt.jsonl"

# scenario constants (tools/make_demo_video.py): the leader brakes at
# CRASH_T, the chaser rear-ends it, traffic queues until the wreck is
# towed at TOW_T, and a pedestrian crosses in between.
CRASH_T, TOW_T = 4.0, 34.0
PED_FROM, PED_TO = 22.0, 30.0


def _load_gt(path: Path) -> dict[int, list]:
    gt: dict[int, list] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            gt[rec["frame"]] = rec["dets"]
    return gt


def run_pipeline(clip: Path, gt: dict[int, list]) -> list:
    """The live loop minus rendering: capture frames, ingest
    detections, let the engine decide. Returns every fired event."""
    src = open_source(str(clip))
    engine = EventEngine(EngineConfig(), frame_shape=(src.height, src.width))
    fired = []
    t_last = 0.0
    try:
        for frame in src.frames():
            fired += engine.update(frame.t, gt.get(frame.index, []))
            t_last = frame.t
    finally:
        src.close()
    # candidates still alive at the end of the stream are confirmed by
    # the runner, so the test must flush them the same way
    fired += engine.finalize(t_last)
    return fired


@pytest.fixture(scope="module")
def demo_events() -> list:
    if not CLIP.exists() or not GT.exists():
        pytest.skip("demo clip or ground truth not present")
    return run_pipeline(CLIP, _load_gt(GT))


def _of(events, type_: str):
    return [e for e in events if e.type == type_]


def test_free_flow_is_silent(demo_events):
    # 0-4s is clean free flow: nothing may fire before the crash
    assert all(e.t > CRASH_T for e in demo_events)


def test_crash_is_reported_once_with_the_right_vehicles(demo_events):
    accidents = _of(demo_events, "ACCIDENT")
    assert len(accidents) == 1
    ev = accidents[0]
    # the chaser (track 2) rear-ends the lane leader (track 1)
    assert set(ev.track_ids) == {1, 2}
    # the generator parks the chaser at a resting overlap of IoU ~0.41
    assert "IoU=0.4" in ev.detail
    # it cannot fire before the impact, and confirmation (0.6s of
    # persistent overlap + 1.5s window) delays it past the impact
    # second - that cost is the point of the retrospective window
    assert CRASH_T < ev.t <= CRASH_T + 4.0


def test_queue_behind_the_crash_is_reported(demo_events):
    jams = _of(demo_events, "JAM")
    origins = _of(demo_events, "JAM_ORIGIN")
    assert len(jams) == 1 and len(origins) == 1
    # the queue forms right after the crash; 4s of persistence plus
    # the 1s confirmation window put the report a few seconds later
    assert CRASH_T < origins[0].t <= CRASH_T + 8.0
    assert jams[0].t == origins[0].t


def test_pedestrian_conflict_fires_while_crossing(demo_events):
    peds = _of(demo_events, "PED_CONFLICT")
    assert len(peds) == 1
    assert PED_FROM <= peds[0].t <= PED_TO + 1.0


def test_no_false_stopped_vehicle(demo_events):
    # the wreck stands still for 30s, but inside a jam - a STOPPED
    # event here would mean the engine lost the plot
    assert not _of(demo_events, "STOPPED")
