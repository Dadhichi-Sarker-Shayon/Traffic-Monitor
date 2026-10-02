# Traffic-Monitor

Realtime traffic observer built on **OpenPSG** (PSGTR, ECCV'22). Point it at a
video file, a webcam or an RTSP stream and it writes back an annotated video and
a machine-readable event log: tracked vehicles with IDs and speeds, panoptic
scene-graph masks and relations, congestion zones, and traffic incidents
(accident, jam, jam origin, queue front, stopped vehicle, pedestrian conflict).

```
my_road.mp4 / webcam / rtsp://...
        │
        ├── YOLOv8 + ByteTrack ──────────── vehicles & persons, persistent IDs, speeds
        ├── OpenPSG (PSGTR-R50, ECCV'22) ── panoptic masks + relations, worker thread
        │        "car driving on road" · "person crossing" · road/pavement/sky stuff
        ├── EventEngine ─────────────────── jam / jam origin / queue front / accident /
        │                                   stopped vehicle / pedestrian conflict
        └── Overlay + JSONL ─────────────── everything burned into the output video
```

OpenPSG runs on every N-th frame in a background thread, so monitoring stays
responsive while the scene graph updates. The decision layer is pure logic —
no GPU, no model — and is unit-tested on its own.

---

## Contents

- [Quickstart](#quickstart)
- [What it detects](#what-it-detects)
- [Verified runs](#verified-runs)
- [CLI reference](#cli-reference)
- [Repository layout](#repository-layout)
- [Environment](#environment)
- [Testing](#testing)
- [Known limitations](#known-limitations)
- [Credits](#credits)

---

## Quickstart

```bash
git clone --recurse-submodules https://github.com/Dadhichi-Sarker-Shayon/Traffic-Monitor.git
cd Traffic-Monitor/realtime-video-tracking

# your own footage / camera / IP cam
python -m traffic_watch --source my_road.mp4 --out outputs/my_road.mp4
python -m traffic_watch --source 0 --show              # webcam, live window
python -m traffic_watch --source rtsp://... --show     # IP camera
```

Outputs land next to each other: annotated `*.mp4` + `<name>.events.jsonl`
(one JSON object per event and per analyzed scene-graph frame — `t`, `frame`,
`type`, `detail`, `zone`, `track_ids`).

Try it without any footage of your own — a synthetic 40s clip with a crash at
4s, a jam, a pedestrian crossing and a stalled vehicle:

```bash
python tools/make_demo_video.py
python -m traffic_watch --source sample_media/clips/demo_traffic.mp4 \
    --detections sample_media/clips/demo_gt.jsonl --no-psg --out outputs/demo_nopsg.mp4
```

## What it detects

Every rule uses hysteresis (on/off thresholds) plus a persistence requirement,
so nothing fires on a single noisy frame.

| event | rule |
|---|---|
| `JAM` | zone with ≥3 vehicles and median speed < 18 px/s, held ≥4s |
| `JAM_ORIGIN` | jammed zone whose onset is clearly earliest → where the queue started |
| `JAM_FRONT` | adjacent zone still flowing → the head of the queue |
| `ACCIDENT` | two vehicles overlapping after fast motion, **held ≥0.6s**; or zone speed collapse (≥65%) with a stable vehicle count (queue growth excluded) |
| `STOPPED` | vehicle stationary ≥8s in an *active* flow — suppressed inside jams and nose-to-tail queues |
| `PED_CONFLICT` | person in a roadway zone with ≥2 vehicles, evidence accumulated ≥1s |

Scene-graph relations from PSGTR (`driving on`, `parked on`, `crossing`, …) are
logged next to the events as `kind:"psg"` rows, so the semantic layer is
inspectable even when no incident fires.

The persistence requirement on `ACCIDENT` is not cosmetic. In dense traffic
seen from above, neighbouring cars overlap for a frame or two constantly —
detector box jitter, tracker ID swaps — and firing on the first overlapping
frame produced two phantom accidents on the aerial clip. Requiring the overlap
to hold removed both.

## Verified runs

Measured on an NVIDIA GTX 1650 (4 GB).

| clip | frames | result |
|---|---|---|
| `accident_intersection.mp4` — busy junction, real crash | 833 | 9 events: accident (IoU 0.89 @1.5s), 3× jam each with its origin, queue front, stopped vehicle @15.5s; 32 scene-graph frames |
| `tiltshift_traffic.mp4` — high-angle, 7–12 vehicles/frame | 1770 | 70 scene-graph frames, **0 events** — traffic keeps flowing, so there is nothing to report. 2.7 fps |
| `demo_traffic.mp4` — synthetic, known ground truth | 40s | accident @4.7s (IoU 0.41), jam + origin @9.9s, pedestrian conflict @27s, no false positives |
| `accident_street.mp4` — street level, sparse traffic | 721 | 25 scene-graph frames, 0 events — only 0.6 vehicles/frame, below the density the jam and collision rules need |
| `aerial_traffic.mp4` — true top-down drone | 1452 | 56 scene-graph frames, **0 vehicles tracked** — the annotated output shows the empty result |
| `drone_traffic.mp4` — very high altitude | 375 | 12 scene-graph frames, **0 vehicles tracked** |

Every clip in `sample_media/clips/` has a matching annotated video and event log
in `outputs/`, so each claim above can be checked without re-running anything.
The two aerial outputs are worth watching specifically: they are the visual
counterpart to the limitation below — the overlays render, the HUD and zones
draw, and not a single track box appears.

Sample scene graph from the high-angle run:

```
car   -[driving on]-> road  0.91      car     -[beside]->     car      0.84
car   -[parked on]-> road  0.85      person  -[carrying]->   backpack 0.82
bus   -[driving on]-> road  0.90      person  -[walking on]-> road     0.89
```

The annotated videos and their event logs are checked in under
`realtime-video-tracking/outputs/` so every claim above can be verified without
re-running anything.

## CLI reference

| flag | default | meaning |
|---|---|---|
| `--source` | *required* | video file, camera index, or `rtsp://` URL |
| `--out` | – | annotated output video |
| `--log` | alongside `--out` | JSONL event log |
| `--detections` | – | ground-truth JSONL, skips YOLO (deterministic runs) |
| `--no-psg` | off | disable the OpenPSG scene-graph layer |
| `--psg-interval N` | 30 | run OpenPSG every N frames (`0` = off) |
| `--conf` | 0.35 | YOLO confidence threshold |
| `--device` | `cuda:0` | `cuda:0` or `cpu` |
| `--max-frames N` | – | stop after N frames |
| `--num-rel` | 12 | relations kept per scene-graph frame |
| `--grid-cols` / `--grid-rows` | 6 / 3 | congestion zone grid |
| `--px-per-meter` | – | calibration: report km/h instead of px/s |
| `--show` | off | live preview window |

## Repository layout

```
realtime-video-tracking/
  traffic_watch/
    capture.py    file | webcam | rtsp source abstraction
    detector.py   YOLOv8 + ByteTrack wrapper (COCO vehicles/persons)
    psg.py        OpenPSG PSGTR adapter (mmdet 2.x) + threaded runner
    events.py     EventEngine — all decision heuristics (pure, tested)
    overlays.py   zones, masks, relation edges, tracks, event banners
    __main__.py   CLI
  tools/          demo-clip generator, YOLO and OpenPSG smoke tests
  tests/          event-engine unit tests
  sample_media/   source clips
  outputs/        annotated videos + event logs
third_party/
  openpsg/        OpenPSG (git submodule, pinned to 34b2a89)
```

## Environment

Verified on Python 3.10 with two virtualenvs on a local disk (the project
itself lives on a synced drive, where virtualenvs do not work):

| env | contents |
|---|---|
| `openpsg1x` | torch 1.13.1+cu117, torchvision 0.14.1, mmcv_full 1.7.1, mmdet 2.28.2, ultralytics 8.4.171, opencv-python |
| `openpsg-env` | torch 2.1.2+cu121 sandbox, used for the unit tests |

Model weights are not committed. Fetch the PSGTR checkpoint into
`third_party/openpsg/work_dirs/checkpoints/epoch_60.pth` from the official
OpenPSG model zoo; the expected SHA256 is
`1c4ddcbda74686568b7e6b8145f7f33030407e27e390c37c23206f95c51829ed`.

Throughput is dominated by OpenPSG: roughly 1.3–2.8 s per analyzed frame on a
GTX 1650, so `--psg-interval` is the main knob between latency and cost.

## Testing

```bash
cd realtime-video-tracking
python -m pytest tests/test_events.py -q      # 9 passed
```

The suite feeds synthetic detection sequences into the event engine — jams that
grow versus jams that form, hysteresis clearing, empty frames, and a regression
test for the transient-overlap accident case — with no GPU or model required.

## Known limitations

- **True top-down drone footage does not work.** The COCO-trained detector
  cannot see cars from directly overhead; this is a domain gap, not a tuning
  problem. Measured on `sample_media/clips/aerial_traffic.mp4` (1452 frames):
  at conf 0.35 / imgsz 640 there are zero vehicles on every sampled frame, and
  at conf 0.05 / imgsz 1280 only junk labels (`suitcase`, `cell phone`,
  `boat`, `airplane`). `drone_traffic.mp4` is higher altitude still. Fixing
  this needs aerial-trained weights (DOTA / VisDrone) or tiling plus rotation
  for small objects. Both clips are kept in the repo so the finding stays
  reproducible.
- Events are geometric and heuristic. A collision is inferred from sustained
  overlap after fast motion, not from visual damage — it will not catch a crash
  where the vehicles separate cleanly, and dense parking can still look
  accident-like.
- The rules need traffic density to work on: a jam needs ≥3 vehicles in a zone
  and a collision needs two overlapping ones, so sparse scenes stay silent by
  design. `accident_street.mp4` is the example — a real street scene that the
  detector only finds 0.6 vehicles per frame in.
- Speeds are in pixels per second unless `--px-per-meter` calibration is given.
- The OpenPSG dependency chain (mmcv 1.x / mmdet 2.x / torch 1.13) is pinned to
  an older CUDA stack; newer GPUs need a rebuild of those wheels.

## Credits

- **OpenPSG / PSGTR** — Zhou et al., *Panoptic Scene Graph Generation*, ECCV 2022.
  Submodule pinned to `Jingkang50/OpenPSG` @ `34b2a89`.
- **Ultralytics YOLO** and **ByteTrack** for detection and multi-object tracking.