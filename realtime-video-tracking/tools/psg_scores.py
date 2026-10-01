"""Print raw score distributions right before OpenPSG's triplet thresholds."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "third_party" / "openpsg"))

import openpsg.models  # noqa: E402,F401
from mmcv import Config  # noqa: E402
from mmdet.apis import inference_detector, init_detector  # noqa: E402

from traffic_watch.psg import CLASSES, PREDICATES, DEFAULT_CFG, DEFAULT_CKPT  # noqa: E402


def main():
    from openpsg.models.relation_heads.psgtr_head import PSGTrHead

    orig = PSGTrHead._get_bboxes_single

    def probe(self, s_cls_score, o_cls_score, r_cls_score, *args, **kwargs):
        with torch.no_grad():
            s_logits = F.softmax(s_cls_score, dim=-1)[..., :-1]
            o_logits = F.softmax(o_cls_score, dim=-1)[..., :-1]
            s_scores, s_labels = s_logits.max(-1)
            o_scores, o_labels = o_logits.max(-1)
            r_lgs = F.softmax(r_cls_score, dim=-1)
            r_logits = r_lgs[..., 1:]
            top_s = s_scores.topk(min(8, s_scores.numel()))
            top_o = o_scores.topk(min(8, o_scores.numel()))
            flat_r = r_logits.flatten()
            top_r = flat_r.topk(min(8, flat_r.numel()))
            print("== score probe ==")
            print(" s_scores top:", [round(v, 3) for v in top_s.values.tolist()])
            print("   their labels:", [CLASSES[int(s_labels[i])] for i in top_s.indices.tolist()],
                  "(0-based CLASSES idx = label-1... showing CLASSES[label])")
            print(" o_scores top:", [round(v, 3) for v in top_o.values.tolist()])
            print("   their labels:", [CLASSES[int(o_labels[i])] for i in top_o.indices.tolist()])
            print(" r_scores top:", [round(v, 3) for v in top_r.values.tolist()])
            print("   their predicates:", [PREDICATES[int(i) % len(PREDICATES)] for i in top_r.indices.tolist()])
            print(f" counts >0.5: s={(s_scores > 0.5).sum().item()}/{s_scores.numel()} "
                  f"o={(o_scores > 0.5).sum().item()}/{o_scores.numel()} "
                  f"r={(flat_r > 0.3).sum().item()}/{flat_r.numel()}")
            print(" s_cls_score shape:", tuple(s_cls_score.shape),
                  "r_cls:", tuple(r_cls_score.shape),
                  "num predicates(self):", getattr(self, "num_relations", "?"),
                  "object_classes len:", len(self.object_classes),
                  "first5:", list(self.object_classes)[:5],
                  "pred first3:", list(self.predicate_classes)[:3])
        return orig(self, s_cls_score, o_cls_score, r_cls_score, *args, **kwargs)

    PSGTrHead._get_bboxes_single = probe

    if len(sys.argv) > 1:  # explicit image path (real photos, etc.)
        frame = cv2.imread(sys.argv[1])
        assert frame is not None, sys.argv[1]
        print("image:", sys.argv[1], frame.shape)
    else:
        video = ROOT / "sample_media" / "clips" / "demo_traffic.mp4"
        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
        ok, frame = cap.read()
        cap.release()
        assert ok

    model = init_detector(Config.fromfile(str(DEFAULT_CFG)), str(DEFAULT_CKPT),
                          device="cuda:0")
    model.eval()
    with torch.no_grad():
        raw = inference_detector(model, frame)
    print("result instances:", len(getattr(raw, "labels", [])),
          "relations:", len(getattr(raw, "rel_pair_idxes", [])))


if __name__ == "__main__":
    main()
