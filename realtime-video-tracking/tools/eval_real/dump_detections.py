import sys, os, json, glob
sys.path.insert(0, r"G:\My Drive\Relate anything\realtime-video-tracking")
from traffic_watch.capture import open_source
from traffic_watch.detector import Detector
from traffic_watch.motion import MotionMeter

MAXS = 60.0
det = Detector(weights=r"G:\My Drive\Relate anything\realtime-video-tracking\yolov8s.pt",
               device="cuda:0", conf=0.55, iou=0.6, imgsz=960)
for clip in sorted(glob.glob(r"D:\traffic_eval\clips\*.mp4")):
    name = os.path.splitext(os.path.basename(clip))[0]
    out = rf"D:\traffic_eval\dets2\{name}.jsonl"
    if os.path.exists(out):
        continue
    try:
        src = open_source(clip)
    except Exception as e:
        print("SKIP", name, e); continue
    det.model.predictor = None  # fresh tracker per clip
    meter = MotionMeter(); n = 0
    with open(out + ".tmp", "w") as f:
        for frame in src.frames():
            if frame.t > MAXS: break
            dets = [d.__dict__ for d in det.track(frame.image)]
            meter.attach(frame.image, dets)
            f.write(json.dumps({"frame": frame.index, "t": frame.t, "dets": dets}) + "\n"); n += 1
    os.replace(out + ".tmp", out)
    json.dump({"w": src.width, "h": src.height, "fps": src.fps, "frames": n}, open(out.replace(".jsonl", ".meta.json"), "w"))
    print("done", name, src.width, src.height, round(src.fps,1), n, flush=True)
