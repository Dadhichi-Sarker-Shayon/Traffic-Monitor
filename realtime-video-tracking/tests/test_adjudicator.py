"""Tests for traffic_watch.adjudicator — no model needed (fake pipe)."""

import numpy as np

from traffic_watch.adjudicator import (
    QUESTIONS,
    VLMAdjudicator,
    crop_event,
)
from traffic_watch.events import Event


class FakePipe:
    def __init__(self, answer):
        self.answer = answer
        self.prompts = []
        self.images = []

    def __call__(self, image, prompt=None):
        self.prompts.append(prompt)
        self.images.append(image)
        return [{"generated_text": self.answer}]


def make_event(type_="ACCIDENT", bbox=(100, 100, 200, 200)):
    return Event(type=type_, t=1.0, bbox=bbox, track_ids=(1, 2))


def make_image():
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def test_agrees_when_model_says_yes():
    adj = VLMAdjudicator(model_id="fake")
    adj._pipe = FakePipe("yes")
    v = adj.verify(make_image(), make_event())
    assert v.agrees
    assert not v.passthrough
    assert adj.agreements == 1
    assert adj.calls == 1


def test_vetoes_when_model_says_no():
    adj = VLMAdjudicator(model_id="fake")
    adj._pipe = FakePipe("no")
    v = adj.verify(make_image(), make_event())
    assert not v.agrees
    assert adj.vetoes == 1


def test_unparseable_answer_passes_through():
    adj = VLMAdjudicator(model_id="fake")
    adj._pipe = FakePipe("the vehicles are blue")
    v = adj.verify(make_image(), make_event())
    assert v.agrees
    assert v.passthrough
    assert adj.passthroughs == 1


def test_unavailable_model_passes_through(monkeypatch):
    adj = VLMAdjudicator(model_id="no-such-model")
    monkeypatch.setattr(adj, "_ensure_pipe", lambda: False)
    adj.last_error = "boom"
    v = adj.verify(make_image(), make_event())
    assert v.agrees
    assert v.passthrough
    assert "unavailable" in v.note


def test_verify_never_raises_on_pipe_error(monkeypatch):
    adj = VLMAdjudicator(model_id="fake")

    def boom(image_rgb, question):
        raise RuntimeError("GPU gone")

    adj._pipe = boom
    v = adj.verify(make_image(), make_event())
    assert v.agrees
    assert v.passthrough


def test_crop_clamps_to_frame_bounds():
    img = make_image()
    ev = make_event(bbox=(-50, -50, 3000, 3000))
    crop = crop_event(img, ev)
    assert crop.shape[0] <= img.shape[0]
    assert crop.shape[1] <= img.shape[1]
    assert crop.shape[2] == 3


def test_crop_with_no_bbox_returns_whole_frame():
    img = make_image()
    ev = make_event(bbox=None)
    crop = crop_event(img, ev)
    assert crop.shape == (img.shape[0], img.shape[1], 3)


def test_every_event_type_has_a_question():
    assert set(QUESTIONS) == {
        "JAM", "JAM_ORIGIN", "JAM_FRONT",
        "ACCIDENT", "STOPPED", "PED_CONFLICT",
    }


def test_unknown_event_type_passes_through():
    adj = VLMAdjudicator(model_id="fake")
    adj._pipe = FakePipe("yes")
    v = adj.verify(make_image(), make_event(type_="SOMETHING_ELSE"))
    assert v.agrees
    assert v.passthrough
    assert adj.calls == 0  # never asked the model


# ----------------------------------------------------------------- scout
from traffic_watch.adjudicator import VLMScout, SCENE_QUESTIONS


class ScriptedPipe:
    def __init__(self, answers):
        self.answers = list(answers)
        self.asked = []

    def __call__(self, image, prompt=None):
        self.asked.append(prompt)
        return [{"generated_text": self.answers.pop(0)}]


def _scout(answers, **kw):
    adj = VLMAdjudicator(model_id="fake")
    adj._pipe = ScriptedPipe(answers)
    return VLMScout(adj, interval_s=1.0, cooldown_s=10.0, kinds=("JAM",), **kw), adj


def test_scout_needs_two_consecutive_yes():
    sc, _ = _scout(["yes", "no", "yes"])
    img = make_image()
    assert sc.scan(img, 0.0) == []
    assert sc.scan(img, 1.0) == []
    assert sc.scan(img, 2.0) == []           # yes after a no restarts the streak


def test_scout_fires_after_two_yes_then_cools_down():
    sc, _ = _scout(["yes", "yes", "yes", "yes"])
    img = make_image()
    assert sc.scan(img, 0.0) == []
    ev = sc.scan(img, 1.0)
    assert len(ev) == 1 and ev[0].type == "JAM"
    assert ev[0].evidence["source"] == "vlm"
    assert sc.scan(img, 2.0) == []           # cooldown
    assert sc.scan(img, 3.0) == []


def test_scout_samples_on_its_interval_only():
    sc, adj = _scout(["no", "no"])
    img = make_image()
    sc.scan(img, 0.0)
    sc.scan(img, 0.3)
    sc.scan(img, 0.6)
    assert adj.calls == 1


def test_scout_skips_kinds_the_engine_already_reports():
    sc, adj = _scout(["yes", "yes"])
    img = make_image()
    assert sc.scan(img, 0.0, skip={"JAM"}) == []
    assert adj.calls == 0


def test_scout_survives_a_missing_model():
    adj = VLMAdjudicator(model_id="no-such-model")
    adj._ensure_pipe = lambda: False
    sc = VLMScout(adj, interval_s=1.0, kinds=("JAM", "ACCIDENT"))
    assert sc.scan(make_image(), 0.0) == []


def test_scene_questions_cover_what_the_scout_asks():
    assert {"JAM", "ACCIDENT"} <= set(SCENE_QUESTIONS)


def test_scene_events_are_judged_on_the_whole_frame():
    adj = VLMAdjudicator(model_id="fake")
    pipe = FakePipe("yes")
    adj._pipe = pipe
    adj.verify(make_image(), make_event("JAM", bbox=(100, 100, 150, 150)))
    adj.verify(make_image(), make_event("PED_CONFLICT", bbox=(100, 100, 150, 150)))
    assert pipe.images[0].shape[:2] == (720, 1280)          # full frame
    assert pipe.images[1].shape[0] < 720                    # crop
