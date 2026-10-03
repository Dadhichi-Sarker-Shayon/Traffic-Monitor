# tools/eval_real/

The scripts that produced the real-footage evaluation in `eval/real/` (see `REPORT.md` there).

| script | does |
|---|---|
| `dump_detections.py` | runs YOLO + ByteTrack once over a folder of clips and saves per-frame detections (JSONL) |
| `score_rules.py` | replays the saved detections through `EventEngine` and scores it against the labels |
| `eval_full_system.py` | adds the VLM veto and scout on top (needs a Python with `transformers`; slow) |
| `report_full_system.py` | turns that output into the rules / +veto / +scout comparison table |

**These are the exact scripts that were used and they contain the paths of the machine they were written on**
(`D:\traffic_eval\...`, `G:\My Drive\...`). Edit the constants at the top before running. Clip ids are in
`eval/real/clip_sources.json`; labels in `ground_truth*.json`.
