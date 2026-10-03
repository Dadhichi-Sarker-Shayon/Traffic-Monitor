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
   ├─ VLM adjudicator (optional) ─ second opinion on fired events (--vlm);
   │                                can veto, never invents
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
  adjudicator.py  optional VLM second opinion (verification-only, never raises)
  overlays.py   zones, masks, relation edges, tracks, event banners
  __main__.py   CLI
tools/
  make_demo_video.py   synthetic 40s clip: crash @4s, jam, pedestrian @22-30s
  psg_smoke.py         one-frame OpenPSG check
  yolo_smoke.py        YOLO recall vs ground truth
tests/               event-engine, adjudicator, PSG parsing, live-pipeline tests
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
event / PSG frame: `t`, `frame`, `type`, `detail`, `zone`, `track_ids`,
and for events an `evidence` dict with the facts behind the decision —
IoU, Δv, speeds and peaks for collisions, zone speeds for jams, stop
duration for stopped vehicles, and so on).

Useful flags: `--psg-interval N` (0 = off), `--no-psg`, `--num-rel`,
`--psg-rel-thresh` (minimum relation score), `--psg-classes` (comma list;
default is the traffic-relevant set, empty string = all),
`--grid-cols/--grid-rows/--roi-top` (zone layout), `--px-per-meter`
(calibration → km/h instead of px/s), `--device cpu`.

Trust-tuning flags: `--conf/--iou` (YOLO confidence / NMS IoU),
`--collision-dv` (speed drop a collision overlap must show, px/s,
default 25), `--imgsz` (raise to 1280 for tall/high-res footage so small far vehicles are detected), `--psg-device cpu` (keeps OpenPSG off a 4GB GPU), `--vlm`, `--vlm-scout`, `--vlm-python`, `--vlm-adapter`, `--vlm-model` (see below).

## How decisions are made

| event | rule (all with hysteresis + persistence) |
|---|---|
| `JAM` | ≥8 on-road vehicles of which ≥65% are crawling (< 0.4 body-widths/s of *net* displacement over ~1s — jitter-free and size-invariant; four-wheelers decide when there are enough, since motorbikes filter through queues), held ≥4s; the older per-zone rule (≥3 vehicles, median < 18 px/s) still applies. Overlap-accident detection is suspended while the road is crawling (bumper contact in a queue is normal) |
| `JAM_ORIGIN` | jammed zone whose onset is clearly earliest → the queue's start |
| `JAM_FRONT` | adjacent zone still flowing → where the queue ends |
| `ACCIDENT` | two vehicles overlapping while both just stopped after fast motion **and** the overlap persists ≥0.6s **and** at least one of them decelerated abruptly (Δv ≥ `--collision-dv`, 25 px/s within 0.5s) — overlap without a speed jump is a merge or queue bumper, not an impact; **or** a pile-up: ≥2 vehicles that each lost ≥65% of a recent fast speed abruptly (Δv) and now stand side by side. Both go through the same sustained-hold gate (0.6s + 0.3s confirm) |
| `STOPPED` | vehicle stationary ≥8s in an *active* flow — suppressed inside jams / nose-to-tail queues |
| `PED_CONFLICT` | tall (aspect ≥1.6) person track ≥0.5s old, on the road (PSG road mask), within one body-height of a *moving* vehicle, for ≥0.4s of real time; riders on bikes/motorbikes are excluded and one alert is issued per pedestrian cluster (5s / quarter-frame radius) |

Scene-graph cues (e.g. PSG's `about to hit`, `crossing`) come from the OpenPSG
layer and are logged alongside; the event log's `kind:"psg"` rows carry the
relations for each analyzed frame.

The persistence requirement on `ACCIDENT` is not cosmetic: in dense traffic
shot from above, neighbouring cars overlap for a frame or two all the time
(detector box jitter, tracker ID swaps). Firing on the first overlapping frame
produced two phantom accidents on the aerial clip below; requiring the overlap
to hold for `collision_persist_s` removed both. The Δv requirement removes
the remaining false positive: two cars that gently merge (or a queue bumper
that taps the car ahead) overlap and stop, but neither shows the velocity
discontinuity of an impact.

### VLM (`--vlm`, `--vlm-scout`)

The default model is **Qwen2-VL-2B-Instruct in 4-bit** (about 1.5 GB of VRAM, roughly
4-9 s per question on a GTX 1650). OpenPSG needs the old torch 1.13 stack, which cannot host
a modern `transformers`, so the VLM can run as a **worker process** in another interpreter:

```bash
python -m traffic_watch --source clip.mp4 --vlm --vlm-scout     --vlm-python "C:/Python314/python.exe" --psg-device cpu --imgsz 960
```

Two roles, both off unless asked for:

- **Adjudication (`--vlm`)** - every fired event is shown to the model with a fixed yes/no
  question. ACCIDENT and JAM are judged on the **whole frame** (a tight crop of two distant
  cars hides the road and the wreckage); PED_CONFLICT and STOPPED on a crop. Agreement keeps
  the event, a "no" withdraws it (`engine.retract`, logged as `{"kind":"veto"}`), and no
  opinion (model missing, unparseable answer) passes the event through unchanged.
- **Scout (`--vlm-scout`)** - the one place the model can *create* an event. Every
  `--vlm-scout-interval` seconds (default 4) of video it looks at a clean whole frame and
  asks "is a crash happening?" and "is this road jammed?". Two consecutive yes answers raise
  the event (`evidence.source = "vlm"`), then 20 s cooldown; kinds the engine already reports
  are skipped.

The model is always shown the **clean** frame, never the one with overlays burned in.

End of run prints `[vlm] N checked, N confirmed, N vetoed, N passthrough`.

## Fine-tuning (Kaggle notebooks)

`notebooks/` has three self-contained Kaggle notebooks (YOLO on VisDrone, a QLoRA crash adapter for the VLM, and a fast
crash classifier). See `notebooks/README.md` - including what was and was not tested.

## Dense traffic / high-angle test

`sample_media/clips/tiltshift_traffic.mp4` — 29.5s, 1280x720 @60fps, high-angle
view of a busy multi-lane street, 7–12 vehicles per frame.

```bash
python -m traffic_watch --source sample_media/clips/tiltshift_traffic.mp4 \
    --out outputs/tiltshift_traffic_full.mp4 --psg-interval 24 --device cuda:0
```

Result: 1770 frames in 662s (**2.7 fps** on a GTX 1650), 70 OpenPSG scene-graph
frames, **0 events** — correct behaviour, the traffic keeps flowing so there is
nothing to report. `outputs/tiltshift_traffic_full.mp4` +
`.events.jsonl` are checked in (the video is re-encoded at crf 28, 72MB→7.8MB,
overlays verified intact afterwards).

Typical scene graph from this clip:

```
car   -[driving on]->  road      0.91      person -[walking on]->   road      0.89
car   -[parked on]->  road      0.85      car     -[beside]->      car       0.84
bus   -[driving on]->  road      0.90      person -[carrying]->    backpack  0.82
```

### Known limit: true top-down aerial footage

The COCO-trained YOLOv8s detector cannot see cars in real top-down drone
footage — a genuine aerial domain gap, not a tuning problem. Measured on
`sample_media/clips/aerial_traffic.mp4` (1452 frames):

| setting | result |
|---|---|
| conf 0.35, imgsz 640 | 0 vehicles, all 7 sampled frames |
| conf 0.05, imgsz 1280 | only junk (`suitcase`, `cell phone`, `boat`, `airplane`) |

`drone_traffic.mp4` is higher altitude still (only rooftops and snow visible).
Supporting aerial weights (e.g. DOTA/VisDrone-trained) or tiling+rotation for
tiny objects would be needed before high-altitude drone streams are usable;
the clips are kept in the repo so the finding stays reproducible.
