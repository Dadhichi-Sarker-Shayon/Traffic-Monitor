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
