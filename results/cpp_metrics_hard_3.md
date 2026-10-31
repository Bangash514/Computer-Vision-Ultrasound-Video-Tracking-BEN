| Tracker | mean error (px) | median (px) | max (px) | success IoU>=0.5 (%) | lost frames (>20 px) | mean error visible (px) | mean error hidden (px) | ms/frame |
|---|---|---|---|---|---|---|---|---|
| Template + Kalman (this repo) | 0.31 | 0.28 | 0.83 | 100.00 | 0 | 0.32 | 0.30 | 0.28 |
| Plain template matching | 0.44 | 0.44 | 1.08 | 100.00 | 0 | 0.43 | 0.52 | 0.20 |
| OpenCV MIL | 34.54 | 55.03 | 71.71 | 39.80 | 180 | 33.64 | 47.07 | 66.19 |
