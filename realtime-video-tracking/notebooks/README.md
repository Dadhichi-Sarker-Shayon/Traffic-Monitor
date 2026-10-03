# Kaggle fine-tuning notebooks

Three independent notebooks. Each one downloads its data, trains, **measures the result against a baseline**, and
writes everything you need to `/kaggle/working/outputs/` (and a `.zip` of it).

| notebook | trains | data | GPU time | main output |
|---|---|---|---|---|
| `01_yolo_finetune_traffic.ipynb` | YOLOv8s detector | VisDrone2019-DET (auto-download, ~2 GB) | ~4-6 h (30 epochs) | `yolov8s_traffic_best.pt` |
| `02_vlm_crash_qlora.ipynb` | Qwen2-VL-2B crash recogniser (QLoRA) | Kaggle crash/no-crash image datasets | ~1.5-2.5 h | `qwen2vl_crash_lora/` |
| `03_crash_classifier_fast.ipynb` | EfficientNet-B0 crash classifier | same datasets as 02 | ~20-40 min | `crash_classifier_ts.pt` + config |

## Run on Kaggle
1. New notebook -> **File -> Import notebook** -> upload the `.ipynb`.
2. Settings: Accelerator **GPU T4 x2** (or P100), **Internet On**.
3. First run with `QUICK = True` (top settings cell, ~3-10 min) to check everything works, then set it to `False`.
4. **Save Version -> Save & Run All (Commit)** so the outputs are kept; download from the **Output** tab.

## Using the results in this project
```bash
# 01 - detector (drop-in: class ids match COCO, nothing else to change)
python -m traffic_watch --source clip.mp4 --weights yolov8s_traffic_best.pt --imgsz 960
# 02 - VLM adapter
python -m traffic_watch --source clip.mp4 --vlm --vlm-scout --vlm-python "C:/Python314/python.exe" \
       --vlm-adapter path/to/qwen2vl_crash_lora
# 03 - the classifier is standalone: see HOW_TO_USE_CLASSIFIER.txt (TorchScript, ~30 ms on a GPU, ~130 ms on CPU)
```

## Kaggle's 12-hour limit
Every notebook keeps a session clock and **stops itself with a margin** (an 11 h deadline, minus time reserved for the final steps):
- **01 YOLO** stops after the epoch it is on if another epoch would not fit (it does *not* use Ultralytics' `time=`
  option, which stretches training to fill the whole budget). `last.pt` is copied to `outputs/` after every epoch.
- **02 VLM** caps training time (`TIME_LIMIT_MIN`, and the deadline minus the time the second scoring pass needs) and writes a
  checkpoint every `CKPT_EVERY_MIN` minutes. Training stops with a clear error if the loss diverges (20 non-finite losses).
- **03 classifier** stops between epochs if time is short and keeps the best epoch.

**If a session is cut off:** commit/attach the earlier run's output as data, then set `RESUME_FROM` (01: the `last.pt`;
02: the `outputs/ckpt` folder; 02 also resumes by itself if `outputs/ckpt` is still there). A YOLO run that *finished* or
was stopped by the time guard cannot be resumed - the notebook refuses it with a message; to train further, set `BASE`
to its `best.pt` and start a new run. Also remember Kaggle's weekly GPU quota (about 30 h).

## Data protocol and honest evaluation (02 and 03)
Needs **two** crash datasets (the notebooks download both when Internet is on):
- **Train:** all of dataset 1 (`suryaprabhakaran2005/road-accidents-from-cctv-footages-dataset`).
- **Tune thresholds:** one half of dataset 2 (`ckay16/accident-detection-from-cctv-footage`) = `val`.
- **Final numbers:** the other half of dataset 2 = `test_EXTERNAL`; it is never used for tuning.
- Near-copies (perceptual hash) of earlier data are removed from each evaluation set, and the notebook warns if a set is
  small or overlaps heavily. Dataset 1 is made of near-identical video frames, so it cannot give a fair test itself.
- With only **one** dataset the notebooks fall back to a leaky split and say so loudly - treat those numbers as optimistic.
- Dataset 2 has no video ids in its file names, so its two halves may still share a few scenes; the test numbers can be
  slightly optimistic. Both datasets are fixed CCTV stills - do not expect the result to carry over to dashcam footage.

## What has and has not been tested
**Tested locally** (fake data shaped like the real datasets, tiny settings): every code cell of all three notebooks
runs without error, including a forced out-of-time stop, a simulated crash followed by a real resume (01 and 02), the
divergence guard (02) and the duplicate check (60/60 re-compressed copies removed, 0/60 different images removed); the class remap in 01 is exact (87/87 boxes) and the result loads in the project's `Detector`;
02's supervised tokens are exactly `Yes<|im_end|>` / `No<|im_end|>` and its adapter loads in the project's worker
process; 03 trains, picks thresholds, and exports TorchScript. These tests found and fixed several real bugs.

**Not tested - be aware:**
- **Not run on Kaggle itself, nor on the real datasets.** Kaggle's package versions can differ; if a cell fails,
  the error is most likely a version mismatch (first things to try: re-run the `pip install` cell).
- **Dataset names come from a web search.** I could not open Kaggle's pages, so the crash datasets' folder layouts are
  unverified. The notebooks scan folder names ("accident / non accident / crash / normal ...") and **print what they
  found**; read that printout. If a dataset fails to download, use *Add data* to attach it, or edit `CANDIDATES`.
- **Kaggle may ask for login** for `kagglehub` downloads; attaching the dataset with *Add data* avoids that.
- **The time guards use measured epoch/scoring times, but real Kaggle speeds are unknown** - the first real run will tell you
  how long each notebook actually takes.
- **No accuracy numbers exist yet.** The notebooks print them; nothing here promises an improvement. Judge 02 and 03
  on the `test_EXTERNAL` rows (a *different* dataset from the one trained on), not on `val`.
- **01 is trained on drone footage.** It should help on elevated and far views and may hurt on dashcam views;
  compare against stock `yolov8s.pt` on your own clips before adopting it.
- Datasets labelled only by folder name contain label noise, and a frame classifier (03) sees one picture, not motion.
- VisDrone has no licence file (research use; cite the VisDrone paper).
