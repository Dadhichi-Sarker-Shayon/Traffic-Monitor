"""PSGTR result parsing: the traffic filter.

PSGTR segments all 133 panoptic classes and its relation scores are
uncalibrated, so parse_psgtr_result() keeps only traffic-relevant
instances, only traffic-meaningful predicates, and only relations
above a score floor. These tests pin that behaviour with a synthetic
mmdet-style result - no model, no GPU.
"""

from __future__ import annotations

import numpy as np

from traffic_watch.psg import (
    CLASSES,
    PREDICATES,
    PSGResult,
    parse_psgtr_result,
    road_mask_of,
)

H = W = 8


def _cls_id(label: str) -> int:
    """1-based PSGTR label id for a class name."""
    return CLASSES.index(label) + 1


def _pred_col(predicate: str) -> int:
    """Column in rel_dists for a predicate (0 is background)."""
    return PREDICATES.index(predicate) + 1


def _region(rows: slice, cols: slice) -> np.ndarray:
    m = np.zeros((H, W), dtype=bool)
    m[rows, cols] = True
    return m


class _FakeRaw:
    """Just enough of mmdet's PSGTR output for the parser."""

    def __init__(self, masks, labels, pairs=None, dists=None, pan=None):
        self.masks = masks
        self.labels = labels
        self.rel_pair_idxes = (np.asarray(pairs, dtype=int)
                               if pairs is not None else np.zeros((0, 2), int))
        self.rel_dists = (np.asarray(dists, dtype=float)
                          if dists is not None else np.zeros((0, len(PREDICATES) + 1)))
        self.pan_results = pan


def _parse(raw, **kw) -> PSGResult:
    return parse_psgtr_result(raw, (H, W), t=0.0, frame_index=0,
                              infer_ms=0.0, **kw)


def _top_relation(res: PSGResult):
    assert len(res.relations) == 1
    return res.relations[0]


def test_non_traffic_instances_are_dropped():
    # PSGTR loves to segment trees/grass on traffic footage; the
    # monitor only wants the traffic actors.
    raw = _FakeRaw(
        masks=[_region(slice(0, 4), slice(0, 4)),      # person
               _region(slice(0, 4), slice(4, 8)),      # car
               _region(slice(4, 8), slice(0, 4)),      # tree-merged
               _region(slice(4, 8), slice(4, 8))],     # grass-merged
        labels=[_cls_id("person"), _cls_id("car"),
                _cls_id("tree-merged"), _cls_id("grass-merged")],
    )
    res = _parse(raw)
    assert [i.label for i in res.instances] == ["person", "car"]


def test_road_is_kept_and_the_stuff_mask_survives():
    # the event engine needs the road mask, so "road" must survive
    # the instance filter and the panoptic stuff map must still work
    pan = np.zeros((H, W), dtype=int)
    pan[4:8, :] = CLASSES.index("road")
    raw = _FakeRaw(
        masks=[_region(slice(4, 8), slice(0, 8))],
        labels=[_cls_id("road")],
        pan=pan,
    )
    res = _parse(raw)
    assert [i.label for i in res.instances] == ["road"]
    mask = road_mask_of(res, H, W)
    assert mask is not None and mask.shape == (H, W)
    assert mask[4:8, :].all() and not mask[0:4, :].any()


def test_relation_above_the_floor_is_kept():
    dists = np.zeros((1, len(PREDICATES) + 1))
    dists[0, _pred_col("walking on")] = 0.8
    raw = _FakeRaw(
        masks=[_region(slice(0, 4), slice(0, 4)),
               _region(slice(0, 4), slice(4, 8))],
        labels=[_cls_id("person"), _cls_id("road")],
        pairs=[[0, 1]],
        dists=dists,
    )
    rel = _top_relation(_parse(raw))
    assert rel.predicate == "walking on"
    assert abs(rel.score - 0.8) < 1e-6


def test_relation_below_the_floor_is_dropped():
    dists = np.zeros((1, len(PREDICATES) + 1))
    dists[0, _pred_col("walking on")] = 0.4   # under the 0.55 floor
    raw = _FakeRaw(
        masks=[_region(slice(0, 4), slice(0, 4)),
               _region(slice(0, 4), slice(4, 8))],
        labels=[_cls_id("person"), _cls_id("road")],
        pairs=[[0, 1]],
        dists=dists,
    )
    assert _parse(raw).relations == []


def test_non_traffic_predicate_is_dropped_even_when_confident():
    # a high score cannot rescue a predicate that means nothing for
    # traffic monitoring
    dists = np.zeros((1, len(PREDICATES) + 1))
    dists[0, _pred_col("kissing")] = 0.9
    raw = _FakeRaw(
        masks=[_region(slice(0, 4), slice(0, 4)),
               _region(slice(0, 4), slice(4, 8))],
        labels=[_cls_id("person"), _cls_id("car")],
        pairs=[[0, 1]],
        dists=dists,
    )
    assert _parse(raw).relations == []


def test_relation_to_a_filtered_instance_is_skipped():
    # tree-merged is not kept, so a relation pointing at it must not
    # survive with a remapped (wrong) endpoint
    dists = np.zeros((1, len(PREDICATES) + 1))
    dists[0, _pred_col("beside")] = 0.9
    raw = _FakeRaw(
        masks=[_region(slice(0, 4), slice(0, 4)),      # person
               _region(slice(0, 4), slice(4, 8)),      # car
               _region(slice(4, 8), slice(0, 4))],     # tree-merged
        labels=[_cls_id("person"), _cls_id("car"),
                _cls_id("tree-merged")],
        pairs=[[2, 1]],   # tree-merged beside car
        dists=dists,
    )
    res = _parse(raw)
    assert res.relations == []
    assert [i.label for i in res.instances] == ["person", "car"]


def test_classes_override_selects_exactly_the_requested_set():
    raw = _FakeRaw(
        masks=[_region(slice(0, 4), slice(0, 4)),
               _region(slice(4, 8), slice(0, 4))],
        labels=[_cls_id("person"), _cls_id("tree-merged")],
    )
    res = _parse(raw, classes=("tree-merged",))
    assert [i.label for i in res.instances] == ["tree-merged"]
