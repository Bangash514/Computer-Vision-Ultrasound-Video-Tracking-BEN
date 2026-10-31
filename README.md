# Ultrasound Video Tracking (C++ and Python, simulated data)

Author: BEN, Software Engineer · 2026-11-01

Tracks a fetal-head-like structure through a **simulated** ultrasound video: speckle noise, fan-shaped
field of view, probe drift and an acoustic-shadow dropout (frames 120-139). Same algorithm in C++ (OpenCV) and Python.

> Research / portfolio code on simulated data. It uses no patient images, is not a medical device,
> and makes no diagnosis (no measurements, no sex prediction).

## Method
1. A template of the structure is cut from the first frame.
2. Each frame: a constant-velocity Kalman filter predicts the position, normalised cross-correlation searches near it.
3. If the match is poor (signal dropout), the filter coasts on its prediction and a wider search tries to re-acquire.

Baselines: plain template matching (best match every frame) and OpenCV MIL.

## Layout
- `python/make_ultrasound_video.py` – simulator, writes AVI + ground-truth CSV
- `python/track_ultrasound.py` – tracker and metrics (Python)
- `cpp/track_ultrasound.cpp` – tracker and metrics (C++), built with `CMakeLists.txt`

## Run
```bash
pip install numpy opencv-python
python3 python/make_ultrasound_video.py --out hard.avi --gt hard.csv --scenario hard --seed 1
python3 python/track_ultrasound.py --video hard.avi --gt hard.csv --tag hard_1
sudo apt install libopencv-dev cmake g++
cmake -S . -B build && cmake --build build
./build/track_ultrasound --video hard.avi --gt hard.csv --tag hard_1
```

## Measured results (OpenCV 4.6, 300 frames, 320x240)
Hard scenario, seeds 1-3 (C++ and Python gave identical numbers). Success = box IoU >= 0.5; lost = error > 20 px.

| Tracker | mean error (px) | success (%) | lost frames | ms/frame (C++) |
|---|---|---|---|---|
| Template + Kalman | 0.30-0.31 | 100 | 0 | ~0.3 |
| Plain template matching | 0.42-0.44 | 100 | 0 | ~0.2 |
| OpenCV MIL | 26.6-34.5 | ~40 | 178-180 | ~68 |

Tables: `results/*metrics*.md`. Picture: `results/montage_hard_1.png` (green = truth, red = this tracker, yellow = MIL).

## Honest limits
- Simulated data only. Real ultrasound has anatomy, varying speckle and larger motion; real accuracy is unmeasured.
- The simulated shadow is easy: plain template matching also survives it, so the Kalman gain is small (about 0.3 vs 0.43 px).
- Template and Kalman settings were tuned on these videos, so the numbers are optimistic.
- One structure, no scale change, no segmentation.
- Possible next step: run on public fetal ultrasound images (e.g. HC18) for segmentation; check the dataset licence first.

## License
MIT, see `LICENSE`.
