# Real-footage evaluation

Date: 2026-10-03. Two separate sets of YouTube clips (first 60 s of each, downloaded with yt-dlp, not redistributed).

| set | clips | used for |
|---|---|---|
| **development** | 28 (`ground_truth.json`) | finding problems and tuning rules/tracker |
| **held-out** | 23 (`ground_truth_heldout.json`) | final scoring only. Labelled from frames *before* any engine output was seen; never used for tuning |

Labels are mine, made from 12-frame contact sheets (subjective, one labeller). `null` = uncertain, excluded.
Each clip counts once per event type: "did the system fire this event type on this clip".
**Small samples**: 7 crash, 4 jam, 1 pedestrian-conflict positives in the held-out set. Treat numbers as indicative.

OpenPSG was **not** part of these runs (it needs more memory than the 7 GB test machine has at 720p+, and only supplies the road mask).
Without the road mask, curbside parked cars and pavement walkers are not filtered, so STOPPED and PED_CONFLICT are pessimistic.
Scoring code: `tools/eval_real/`. Raw outputs: `heldout_full_system_results.json`, `heldout_rules_events.json`.

## What was changed

| area | change |
|---|---|
| tracker | `track_buffer 30->120`, `match_thresh 0.95`, `new_track_thresh 0.6`: track IDs per concurrent vehicle fell 20-45% on the worst clips |
| jam | scene-level rule: >=8 on-road vehicles, >=65% crawling (<0.4 body-widths/s of *net* displacement over 1 s). Jitter-proof, resolution-independent, four-wheelers decide when there are enough (motorbikes filter through queues) |
| accident | overlap rule suspended while the whole road crawls; contact must start between *established* tracks (>=1.5 s old, >=3 s for pile-ups); raw speeds above 25 body-widths/s (a track ID jumping between cars) are rejected; pile-up needs vehicles that were clearly moving |
| pedestrian | must be on the road, a tall person box (riders excluded), within one body-height of a *moving* vehicle, sustained 0.4 s real time, one alert per cluster; the persistence timer now counts real seconds, not frames |
| VLM | Qwen2-VL-2B (4-bit) works inside the project, in-process or as a worker process; sees the clean frame (it previously got the one covered in overlays); ACCIDENT/JAM judged on the whole frame; new **scout** that can propose a jam/crash from sampled frames |
| bugs | event log closed before end-of-stream events were written; draw-zone memory crash on 4K; ghost boxes drawn for lost tracks |
| tests | 37 -> 58 (jam, pedestrian, pile-up gates, scout) |

## Held-out results (23 clips, final system)

| event | rules only | + VLM veto | + veto + VLM scout |
|---|---|---|---|
| **JAM** | P 0.67 R 0.50 | P 1.00 R 0.50 | **P 1.00 R 1.00** (4/4, 0 false) |
| **ACCIDENT** | P 0.40 R 0.29 | P 1.00 R 0.14 | P 0.67 R 0.29 |
| STOPPED | 0 right, 3 false | unchanged | unchanged |
| PED_CONFLICT | 1/1, 0 false | 1/1, 0 false | 1/1, 0 false |

Accident timing: every accident that was detected fell in the right time window (2/2).
The VLM vetoed 12 of 37 alerts.

### What this says
- **Jams: works on these clips.** The scene-level rule plus the VLM scout found all 4 jams with no false alarm. Rules alone found 2 of 4.
- **Accidents: weak.** Recall is 2 of 7 at best. Found: the pickup crash (rules and scout) and the Pampore crash (scout only). Missed: the toll-booth crash (rules found it, the VLM veto wrongly removed it), a truck rollover, a dashcam T-bone, and two cab-view crashes. The VLM veto trades recall for precision: it removed false alerts but also one real crash.
- **STOPPED is poor** here: it fires on parked cars at the curb and cars waiting at lights. It needs the OpenPSG road mask (not used in this test).
- **PED_CONFLICT**: only one positive clip, so no real conclusion.

## Development set (28 clips), rules only, before -> after the fixes

| event | before (TP/FP/FN) | after |
|---|---|---|
| ACCIDENT | 2 / 4 / 5 | 0 / 0 / 7 |
| JAM | 1 / 0 / 2 | 2 / 0 / 1 |
| STOPPED | 1 / 1 / 1 | 1 / 1 / 1 |
| PED_CONFLICT | 2 / 7 / 1 | 2 / 6 / 1 |

The stricter accident gates removed every false alarm on this set but also the two true detections, which is why the VLM scout matters for accidents. (PED "no conflict" labels here are weak: I marked clips without an obvious pedestrian scene as negative without checking every second.)
Held-out accident precision (0.40, rules only) is in line with the earlier development numbers, so the tuning did not overfit visibly.

## VLM on its own (development set, single frames, Qwen2-VL-2B)

| task | precision | recall | accuracy |
|---|---|---|---|
| crash visible in frame (n=54) | 0.88 | 0.39 | 0.78 |
| jam visible in frame (n=29) | 0.89 | 0.89 | 0.93 |

About 4-9 s per question on a GTX 1650 (slower when RAM is short).

## Conclusion
- **Usable now, for recorded video:** jam detection, with the VLM scout (4/4 on held-out, 0 false alarms; small sample).
- **Not production ready:** accident detection (found 2 of 7 held-out crashes), stopped-vehicle detection (needs the road mask), and anything on a moving camera or cab view.
- **Speed:** about 1-3 fps for the full stack on this machine (VLM scout adds a few seconds per sampled frame); fine for offline analysis, too slow for live cameras.

## Next steps, in order of value
1. Run with OpenPSG on a machine with more RAM and re-score STOPPED and PED_CONFLICT with the road mask.
2. Label 30+ more real crash clips; the accident numbers rest on 7.
3. Reduce wrongful vetoes of real crashes (try a second question, or only veto when the model says no twice).
4. Treat moving-camera and cab-view footage as out of scope, or add ego-motion handling.
