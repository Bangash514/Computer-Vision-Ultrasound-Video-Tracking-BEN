| Tracker | mean error (px) | median (px) | max (px) | success IoU>=0.5 (%) | lost frames (>20 px) | mean error visible (px) | mean error hidden (px) | ms/frame |
|---|---|---|---|---|---|---|---|---|
| Template + Kalman (this repo) | 0.30 | 0.29 | 0.82 | 100.00 | 0 | 0.30 | 0.33 | 0.42 |
| Plain template matching | 0.43 | 0.45 | 1.00 | 100.00 | 0 | 0.43 | 0.49 | 0.29 |
| OpenCV MIL | 34.16 | 53.99 | 61.97 | 39.80 | 180 | 33.63 | 41.53 | 79.64 |
