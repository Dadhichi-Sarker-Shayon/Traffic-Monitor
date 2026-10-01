"""Deep-dive into the raw PSGTR output to find why predictions look wrong."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "third_party" / "openpsg"))

import torch  # noqa: E402

import openpsg.models  # noqa: E402,F401
from mmcv import Config  # noqa: E402
from mmdet.apis import inference_detector  # noqa: E402

from traffic_watch.psg import DEFAULT_CFG, DEFAULT_CKPT  # noqa: E402


def main():
    video = ROOT / "sample_media" / "clips" / "demo_traffic.mp4"
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
    ok, frame = cap.read()
    cap.release()
    assert ok

    cfg = Config.fromfile(str(DEFAULT_CFG))
    print("test pipeline:", [t["type"] for t in cfg.data.test.pipeline])
    print("img_norm:", cfg.get("img_norm_cfg", "n/a"))
    print("test_cfg:", cfg.model.get("test_cfg", "n/a"))

    from mmdet.apis import init_detector
    model = init_detector(cfg, str(DEFAULT_CKPT), device="cuda:0")
    model.eval()
    devs = {n: p.device.type for n, p in list(model.named_parameters())[:3]}
    print("param devices:", devs, "| training:", model.training)

    # time a couple of calls (warmup vs steady)
    for call in range(2):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        raw = inference_detector(model, frame)
        torch.cuda.synchronize()
        print(f"infer#{call}: {(time.perf_counter()-t0)*1000:.0f} ms")

    print("raw type:", type(raw))
    if isinstance(raw, (tuple, list)):
        print("  len:", len(raw))
        raw = raw[0] if len(raw) == 1 else raw
        print("  elem0 type:", type(raw[0]) if len(raw) else None)
    for fld in ("pan_results", "masks", "labels", "rel_pair_idxes", "rel_dists",
                "rel_scores", "triplet_scores", "bbox", "segm"):
        v = getattr(raw, fld, None)
        if v is None:
            print(f"  {fld}: <absent>")
            continue
        if hasattr(v, "shape"):
            print(f"  {fld}: shape={v.shape} dtype={getattr(v, 'dtype', '?')}")
        elif isinstance(v, (list, tuple)):
            print(f"  {fld}: list[{len(v)}] elem0={type(v[0]) if v else None} "
                  f"shape={getattr(v[0], 'shape', None) if v else None}")
        else:
            print(f"  {fld}: {type(v)}")

    labels = getattr(raw, "labels", None)
    if labels is not None:
        labels = np.asarray(labels).reshape(-1)
        print("labels:", labels[:30], "n=", len(labels))
        from traffic_watch.psg import CLASSES
        print("label names:", [CLASSES[int(l) - 1] if 0 <= int(l) - 1 < len(CLASSES) else f"?{int(l)}"
                               for l in labels[:30]])

    masks = getattr(raw, "masks", None)
    if masks is not None and len(masks):
        m = np.asarray(masks)
        print("masks:", m.shape, m.dtype, "areas:", [int(mm.sum()) for mm in m[:10]])

    pan = getattr(raw, "pan_results", None)
    if pan is not None:
        pan = np.asarray(pan)
        u, c = np.unique(pan, return_counts=True)
        print("pan_results uniques (id:count) first 20:", list(zip(u[:20].tolist(), c[:20].tolist())))
        from traffic_watch.psg import CLASSES, INSTANCE_OFFSET
        names = [CLASSES[int(i) % INSTANCE_OFFSET] if int(i) % INSTANCE_OFFSET < len(CLASSES) else "?"
                 for i in u[:20]]
        print("pan names:", names)

    rel_dists = getattr(raw, "rel_dists", None)
    if rel_dists is not None:
        rd = np.asarray(rel_dists)
        print("rel_dists:", rd.shape, "row maxima:", rd.max(axis=1)[:10] if rd.ndim == 2 else "-")
        print("pred argmax:", rd[:, 1:].argmax(axis=1)[:10] if rd.ndim == 2 else "-")

    # 1) weight compatibility: ckpt state_dict vs model state_dict
    ckpt = torch.load(str(DEFAULT_CKPT), map_location="cpu")
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    msd = model.state_dict()
    missing = [k for k in msd if k not in sd]
    unexpected = [k for k in sd if k not in msd]
    mismatch = [k for k in sd if k in msd and tuple(sd[k].shape) != tuple(msd[k].shape)]
    print(f"ckpt keys={len(sd)} model keys={len(msd)} missing={len(missing)} "
          f"unexpected={len(unexpected)} shape-mismatch={len(mismatch)}")
    for k in missing[:8]:
        print("  missing:", k, tuple(msd[k].shape))
    for k in unexpected[:8]:
        print("  unexpected:", k, tuple(sd[k].shape))
    for k in mismatch[:8]:
        print("  mismatch:", k, "ckpt", tuple(sd[k].shape), "model", tuple(msd[k].shape))
    # how well does the checkpoint fit? overlap ratio
    common = [k for k in sd if k in msd and tuple(sd[k].shape) == tuple(msd[k].shape)]
    same = sum(1 for k in common if torch.equal(sd[k].cpu(), msd[k].cpu()))
    print(f"common={len(common)} identical-params={same}")

    # 2) capture the actual input tensor via a backbone hook
    cap = {}

    def pre_hook(module, args):
        cap["x"] = args[0].detach()

    model.backbone.conv1.register_forward_pre_hook(pre_hook)
    with torch.no_grad():
        inference_detector(model, frame)
    x = cap.get("x")
    assert x is not None
    print("backbone input:", tuple(x.shape), x.dtype,
          "min/max:", round(float(x.min()), 2), round(float(x.max()), 2),
          "mean/std:", round(float(x.mean()), 3), round(float(x.std()), 3))
    print("per-ch mean:", [round(float(x[0, c].mean()), 2) for c in range(x.shape[1])],
          "per-ch std:", [round(float(x[0, c].std()), 2) for c in range(x.shape[1])])
    # what would an un-normalized feed look like for comparison?
    raw_img = frame[:, :, ::-1].astype(np.float32)  # BGR->RGB
    print("raw RGB mean/std:", round(float(raw_img.mean()), 2), round(float(raw_img.std()), 2),
          "expected-norm range: [", round((0 - 123.675) / 58.395, 2), ",",
          round((255 - 123.675) / 58.395, 2), "]")


if __name__ == "__main__":
    main()
