# Eval report

7/30 windows labelled (0 real, 7 synthetic).

Headline numbers come from real footage only; the synthetic demo clip is reported separately because it is flat graphics, not camera imagery.

## SYNTHETIC DEMO (7 windows)

### JAM (state)

tp=6 fp=0 fn=1 | precision=1.00 recall=0.86 f1=0.92

Missed (label says yes, engine silent):
- `demo_traffic_000052` — chosen: event:ACCIDENT

### ACCIDENT (instant)

tp=1 fp=0 fn=0 | precision=1.00 recall=1.00 f1=1.00

## Ground truth

- jam windows: 7
- accident windows: 1
- windows with visible damage: 7
- still unlabelled: 23

