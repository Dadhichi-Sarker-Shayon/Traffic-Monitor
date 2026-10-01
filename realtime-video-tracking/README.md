# traffic_watch — OpenPSG realtime traffic observer

Marks **everything that's happening** directly on the video: vehicles with
persistent IDs and speeds, OpenPSG scene-graph masks + relations, congestion
zones, and decisions (accident / jam / jam origin / queue head / stopped
vehicle / pedestrian conflict) as on-frame banners with a machine-readable
event log.

```
video file / webcam / RTSP
   │
   ├─ YOLOv8 + ByteTrack ────────── vehicles & persons, persistent IDs, speeds
   ├─ OpenPSG (PSGTR-r50, ECCV'22) ─ panoptic masks + relations, background thread
   │      "car driving on road", "person crossing", road/pavement/sky stuff masks
   ├─ EventEngine ───────────────── jam / jam origin / queue head / accident /
   │                                stopped vehicle / pedestrian conflict
   └─ Overlay + JSONL log ───────── everything burned into the output video
```

OpenPSG runs on **every N-th frame** in a worker thread (`--psg-interval`,
default 30) so monitoring stays responsive while the scene graph updates in
the background. The event engine is pure logic (unit-tested, no GPU needed).

## Layout

```
traffic_watch/
  capture.py    file | webcam index | rtsp:// source abstraction
  detector.py   YOLOv8 + ByteTrack wrapper (COCO vehicles/persons)
  psg.py        OpenPSG PSGTR adapter (mmdet 2.x) + threaded runner
  events.py     EventEngine: all decision heuristics (pure, tested)
  overlays.py   zones, masks, relation edges, tracks, event banners
  __main__.py   CLI
tools/
  make_demo_video.py   synthetic 40s clip: crash @4s, jam, pedestrian @22-30s
  psg_smoke.py         one-frame OpenPSG check
  yolo_smoke.py        YOLO recall vs ground truth
tests/test_events.py   event-engine unit tests
```

## Environments (Windows)

Two venvs on the local disk (the project itself lives on a synced drive):

| env | path | purpose |
|---|---|---|
| `openpsg1x` | `D:\envs\openpsg1x` | **runs everything** — torch 1.13.1+cu117, mmcv_full 1.7.1, mmdet 2.28.2, ultralytics |
| `openpsg-env` | `D:\envs\openpsg-env` | torch 2.1.2+cu121 sandbox, used for unit tests |

OpenPSG code: `third_party/openpsg` (Jingkang50/OpenPSG, ECCV'22),
checkpoint `third_party/openpsg/work_dirs/checkpoints/epoch_60.pth`
(PSGTR-R50, SHA256 verified against the official model zoo).

## Run

```bash
# 0) generate the demo clip (already checked in? skip)
cd realtime-video-tracking
python tools/make_demo_video.py

# 1) deterministic demo (no YOLO): events + overlays only
python -m traffic_watch --source sample_media/clips/demo_traffic.mp4 \
    --detections sample_media/clips/demo_gt.jsonl --no-psg \
    --out outputs/demo_nopsg.mp4

# 2) full stack: YOLO + OpenPSG + events
python -m traffic_watch --source sample_media/clips/demo_traffic.mp4 \
    --out outputs/demo_full.mp4

# 3) your own footage / camera / IP cam
python -m traffic_watch --source my_road.mp4 --out outputs/my_road.mp4
python -m traffic_watch --source 0 --show            # webcam, live window
python -m traffic_watch --source rtsp://... --show   # IP camera
```

Outputs: annotated `*.mp4` + `<name>.events.jsonl` (one JSON object per
event / PSG frame: `t`, `frame`, `type`, `detail`, `zone`, `track_ids`).

Useful flags: `--psg-interval N` (0 = off), `--no-psg`, `--num-rel`,
`--grid-cols/--grid-rows/--roi-top` (zone layout), `--px-per-meter`
(calibration → km/h instead of px/s), `--device cpu`.

## How decisions are made

| event | rule (all with hysteresis + persistence) |
|---|---|
| `JAM` | zone with ≥3 vehicles and median speed < 18 px/s, held ≥4s |
| `JAM_ORIGIN` | jammed zone whose onset is clearly earliest → the queue's start |
| `JAM_FRONT` | adjacent zone still flowing → where the queue ends |
| `ACCIDENT` | two vehicles overlapping while both just stopped after fast motion, **or** zone speed collapse (≥65%) with stable vehicle count (queue growth excluded) |
| `STOPPED` | vehicle stationary ≥8s in an *active* flow — suppressed inside jams / nose-to-tail queues |
| `PED_CONFLICT` | person in a roadway zone with ≥2 vehicles, evidence accumulated ≥1s |

Scene-graph cues (e.g. PSG's `about to hit`, `crossing`) come from the OpenPSG
layer and are logged alongside; the event log's `kind:"psg"` rows carry the
relations for each analyzed frame.
