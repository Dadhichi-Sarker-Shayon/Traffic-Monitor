"""traffic_watch CLI — run the traffic observer on a video / camera / RTSP.

Example:
    python -m traffic_watch --source sample_media/clips/demo.mp4 --out outputs/demo_annotated.mp4
    python -m traffic_watch --source 0 --show          # webcam
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from .capture import open_source
from .detector import Detector
from .events import EngineConfig, EventEngine
from .motion import MotionMeter
from .overlays import draw_events, draw_psg, draw_tracks, draw_zones, _put_label
from .psg import PSGRunner, road_mask_of


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="traffic_watch", description="OpenPSG-based realtime traffic observer")
    p.add_argument("--source", required=True, help="video file | camera index | rtsp:// URL")
    p.add_argument("--out", default=None, help="annotated output video (mp4)")
    p.add_argument("--log", default=None, help="JSONL event log path (default: alongside --out)")
    p.add_argument("--show", action="store_true", help="live preview window")
    p.add_argument("--device", default="cuda:0", help="inference device (cuda:0 / cpu)")
    p.add_argument("--weights", default="yolov8s.pt", help="YOLO weights")
    p.add_argument("--conf", type=float, default=0.35, help="YOLO confidence")
    p.add_argument("--detections", default=None,
                   help="ground-truth detections JSONL (bypasses YOLO, deterministic demo)")
    p.add_argument("--max-frames", type=int, default=None, help="stop after N frames")
    p.add_argument("--psg-interval", type=int, default=30, help="run OpenPSG every N frames (0=off)")
    p.add_argument("--no-psg", action="store_true", help="disable the OpenPSG scene layer")
    p.add_argument("--num-rel", type=int, default=12, help="relations kept per PSG frame")
    p.add_argument("--px-per-meter", type=float, default=None, help="calibration: px per meter (enables km/h)")
    p.add_argument("--grid-cols", type=int, default=6)
    p.add_argument("--grid-rows", type=int, default=3)
    p.add_argument("--roi-top", type=float, default=0.45, help="ROI top edge as fraction of frame height")
    p.add_argument("--psg-masks", action="store_true", default=True, help="draw PSG masks")
    p.add_argument("--no-psg-masks", dest="psg_masks", action="store_false")
    p.add_argument("--jam-speed-mode", choices=["abs", "auto", "metric"], default="abs",
                   help="abs = fixed px/s (default); auto = relative to this clip's "
                        "traffic speed; metric = m/s via --px-per-meter")
    p.add_argument("--jam-speed-rel-frac", type=float, default=0.25,
                   help="auto mode: congested below this share of the scene speed")
    p.add_argument("--confirm-jam", type=float, default=1.0,
                   help="seconds a jam candidate must survive before it is reported")
    p.add_argument("--confirm-accident", type=float, default=1.5,
                   help="seconds an accident candidate must survive before it is reported")
    p.add_argument("--no-road-mask", dest="road_mask", action="store_false",
                   help="ignore the OpenPSG road mask when counting vehicles in zones")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    src = open_source(args.source, max_frames=args.max_frames)
    fps = src.fps or 30.0
    print(f"[capture] {args.source}  {src.width}x{src.height} @ {fps:.1f}fps  live={src.live}")

    if args.detections:
        gt: dict[int, list] = {}
        with open(args.detections, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                gt[rec["frame"]] = rec["dets"]
        print(f"[detector] ground truth from {args.detections} ({len(gt)} frames)")
        det = None
    else:
        det = Detector(weights=args.weights, device=args.device, conf=args.conf)
    cfg = EngineConfig(grid_cols=args.grid_cols, grid_rows=args.grid_rows, roi_top_frac=args.roi_top,
                       jam_speed_mode=args.jam_speed_mode,
                       jam_speed_rel_frac=args.jam_speed_rel_frac,
                       px_per_meter=args.px_per_meter,
                       confirm_jam_s=args.confirm_jam,
                       confirm_accident_s=args.confirm_accident)
    engine = EventEngine(cfg, frame_shape=(src.height, src.width))
    meter = MotionMeter()

    psg = PSGRunner(
        device=args.device,
        interval=0 if args.no_psg else args.psg_interval,
        num_rel=args.num_rel,
        enabled=not args.no_psg,
    )
    psg.start()

    out_path = Path(args.out) if args.out else None
    if out_path is None and not args.show:
        out_path = Path("outputs") / "traffic_annotated.mp4"
    writer = None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (src.width, src.height))

    log_path = Path(args.log) if args.log else (out_path.with_suffix(".events.jsonl") if out_path else None)
    log_f = open(log_path, "w", encoding="utf-8") if log_path else None
    if log_path:
        print(f"[log] {log_path}")

    def log(obj):
        if log_f:
            log_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    show = args.show
    fps_ema = 0.0
    n = 0
    last_road_mask_frame = -1
    last_frame_t = 0.0
    last_psg_logged = -1
    t_start = time.perf_counter()
    try:
        for frame in src.frames():
            t0 = time.perf_counter()
            if det is None:
                dets = gt.get(frame.index, [])
            else:
                dets = [d.__dict__ for d in det.track(frame.image)]
                meter.attach(frame.image, dets)
            fired = engine.update(frame.t, dets)

            if psg.should_submit(frame.index):
                psg.submit(frame.index, frame.t, frame.image)

            img = frame.image
            res = psg.latest()
            if res is not None and res.frame_index >= 0:
                draw_psg(img, res, show_masks=args.psg_masks, show_relations=True)
                # feed the road region to the engine so "cars standing on the
                # road" stops counting cars parked in a side street
                if args.road_mask and res.frame_index != last_road_mask_frame:
                    road = road_mask_of(res, src.height, src.width)
                    if road is not None:
                        engine.set_road_mask(road)
                        last_road_mask_frame = res.frame_index
            draw_zones(img, engine, cfg)
            draw_tracks(img, engine, px_per_meter=args.px_per_meter)
            draw_events(img, engine, frame.index)

            # HUD
            inst = 1.0 / max(1e-3, time.perf_counter() - t0)
            fps_ema = inst if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst
            hud = f"fps {fps_ema:.1f} | veh {sum(1 for t_ in engine.tracks.values() if t_.cls != 'person')} | psg "
            if not psg.enabled or psg.interval <= 0:
                hud += "off"
            elif res is None:
                hud += "loading..."
            else:
                age = frame.index - res.frame_index
                hud += f"f{res.frame_index} {res.infer_ms:.0f}ms"
                if psg.last_error:
                    hud += f" ERR:{psg.last_error[:40]}"
            _put_label(img, hud, (8, img.shape[0] - 10), (255, 255, 255), scale=0.5, bg=(25, 25, 25))

            for ev in fired:
                log({"kind": "event", "t": round(frame.t, 3), "frame": frame.index,
                     "type": ev.type, "detail": ev.detail, "zone": ev.zone,
                     "track_ids": list(ev.track_ids), "score": round(ev.score, 2)})
                print(f"[event] t={frame.t:7.2f}s  {ev.type:14s} {ev.detail}")
            if res is not None and res.frame_index >= 0 and res.frame_index != last_psg_logged:
                last_psg_logged = res.frame_index
                log({"kind": "psg", "t": round(res.t, 3), "frame": res.frame_index,
                     "instances": [(i.label) for i in res.instances],
                     "relations": [{"s": res.instances[r.s].label, "p": r.predicate,
                                    "o": res.instances[r.o].label, "score": round(r.score, 3)}
                                   for r in res.top_relations(args.num_rel)]})

            if writer is not None:
                writer.write(img)
            if show:
                try:
                    cv2.imshow("traffic_watch", img)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
                except cv2.error:
                    show = False
                    print("[warn] no GUI available, disabling --show")
            n += 1
            last_frame_t = frame.t
    except KeyboardInterrupt:
        print("[stop] interrupted")
    finally:
        psg.close()
        src.close()
        if writer is not None:
            writer.release()
        if log_f:
            log_f.close()
        if show:
            cv2.destroyAllWindows()

    dur = time.perf_counter() - t_start
    # candidates still alive at end of stream: confirm them, or the last
    # `confirm_*` seconds of every clip would silently vanish
    for ev in engine.finalize(last_frame_t):
        log({"kind": "event", "t": round(ev.t, 3), "frame": n, "type": ev.type,
             "detail": ev.detail, "zone": ev.zone, "track_ids": list(ev.track_ids),
             "score": round(ev.score, 2), "note": "confirmed at end of stream"})
        print(f"[event] t={ev.t:7.2f}s  {ev.type:14s} {ev.detail}  (end-of-stream confirm)")
    print(f"[done] {n} frames in {dur:.1f}s ({n / max(dur, 1e-6):.1f} fps)")
    if out_path:
        print(f"[done] annotated video: {out_path}")
    print(f"[done] events fired: {len(engine.history)}")
    if psg.last_error:
        print(f"[psg] last error: {psg.last_error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
