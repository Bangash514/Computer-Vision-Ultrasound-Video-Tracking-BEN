| Tracker | mean error (px) | median (px) | max (px) | success IoU>=0.5 (%) | lost frames (>20 px) | mean error visible (px) | mean error hidden (px) | ms/frame |
|---|---|---|---|---|---|---|---|---|
| Template + Kalman (this repo) | 0.30 | 0.29 | 0.79 | 100.00 | 0 | 0.30 | 0.36 | 0.28 |
| Plain template matching | 0.42 | 0.44 | 1.01 | 100.00 | 0 | 0.42 | 0.50 | 0.22 |
| OpenCV MIL | 26.63 | 34.78 | 68.27 | 40.80 | 178 | 25.93 | 36.42 | 70.35 |
