"""Hook every submodule; report which ones emit NaN/Inf during inference."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "third_party" / "openpsg"))

import openpsg.models  # noqa: E402,F401
from mmcv import Config  # noqa: E402
from mmdet.apis import inference_detector, init_detector  # noqa: E402

from traffic_watch.psg import DEFAULT_CFG, DEFAULT_CKPT  # noqa: E402


def main():
    video = ROOT / "sample_media" / "clips" / "demo_traffic.mp4"
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
    ok, frame = cap.read()
    cap.release()
    assert ok

    model = init_detector(Config.fromfile(str(DEFAULT_CFG)), str(DEFAULT_CKPT),
                          device="cuda:0")
    model.eval()

    bad = {}
    outputs = {}

    def make_hook(name):
        def hook(module, args, out):
            tensors = []
            if torch.is_tensor(out):
                tensors = [out]
            elif isinstance(out, (tuple, list)):
                tensors = [o for o in out if torch.is_tensor(o)]
            elif isinstance(out, dict):
                tensors = [o for o in out.values() if torch.is_tensor(o)]
            for t in tensors:
                if t.is_floating_point():
                    t32 = t.float()
                    if torch.isnan(t32).any() or torch.isinf(t32).any():
                        bad[name] = (float(t32.nan_to_num(0).abs().max()),
                                     bool(torch.isnan(t32).any()),
                                     bool(torch.isinf(t32).any()))
                    outputs[name] = (tuple(t.shape), round(float(t32.mean()), 3),
                                     round(float(t32.std()), 3))
        return hook

    handles = []
    for name, mod in model.named_modules():
        handles.append(mod.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        try:
            raw = inference_detector(model, frame)
        finally:
            for h in handles:
                h.remove()

    print(f"modules scanned: {len(outputs)}")
    if bad:
        print(f"NaN/Inf modules: {len(bad)}")
        for name, (mx, hasnan, hasinf) in list(bad.items())[:25]:
            print(f"  {name or '<root>'}: max|v|={mx:.4g} nan={hasnan} inf={hasinf}")
    else:
        print("NO NaN/Inf detected in any module output")

    # interesting intermediates: anything named head / logits / decoder
    print("\n--- head-ish outputs ---")
    for name, (shape, mean, std) in outputs.items():
        if any(k in name for k in ("head", "logit", "decoder", "cls", "score")):
            print(f"  {name}: shape={shape} mean={mean} std={std}")

    # gradient-free second opinion: panoptic prediction before argmax
    print("\npan unique ids:", np.unique(np.asarray(raw.pan_results))[:10])


if __name__ == "__main__":
    main()
