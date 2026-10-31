# BEN 2026-11-01
# Change aim: Make a SIMULATED ultrasound-like video (speckle, fan shape, a fetal-head-like ellipse) with ground truth.
# File: make_ultrasound_video.py
"""Simulated data only: no patient images. Writes an AVI and a ground-truth CSV (frame,cx,cy,w,h,visible).
Scenarios: clean | hard (extra noise, probe drift, acoustic-shadow dropout in frames 120-139)."""
import argparse
import csv
import numpy as np
import cv2

W, H = 320, 240


# BEN 2026-11-01
# Change aim: Fully developed speckle: Rayleigh-like texture from two blurred Gaussian fields.
# File: make_ultrasound_video.py
# Function: speckle
def speckle(rng, shape):
    a = cv2.GaussianBlur(rng.standard_normal(shape).astype(np.float32), (0, 0), 1.2)
    b = cv2.GaussianBlur(rng.standard_normal(shape).astype(np.float32), (0, 0), 1.2)
    s = np.sqrt(a * a + b * b)
    return s / s.mean()


# BEN 2026-11-01
# Change aim: Fan-shaped (sector) field of view like a curved probe image.
# File: make_ultrasound_video.py
# Function: fan_mask
def fan_mask():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ang = np.degrees(np.arctan2(xx - W / 2, yy + 20))
    rad = np.hypot(xx - W / 2, yy + 20)
    return ((np.abs(ang) < 38) & (rad < H + 15)).astype(np.float32)


# BEN 2026-11-01
# Change aim: Generate the video and the ground truth for one scenario and seed.
# File: make_ultrasound_video.py
# Function: main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--scenario", default="hard", choices=["clean", "hard"])
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    hard = a.scenario == "hard"
    pad = 60
    tissue = speckle(rng, (H + 2 * pad, W + 2 * pad)) * 0.55      # background tissue (static pattern)
    obj_tex = speckle(rng, (140, 140))                              # texture that travels with the object
    mask = fan_mask()
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"MJPG"), 25, (W, H), False)
    rows = []
    ax, ay = 36.0, 27.0                                             # ellipse half axes
    for i in range(a.frames):
        cx = 160 + 60 * np.sin(2 * np.pi * i / 240)
        cy = 120 + 35 * np.sin(2 * np.pi * i / 150 + 1)
        ang = 12 * np.sin(2 * np.pi * i / 200)
        drift = (3 * np.sin(2 * np.pi * i / 90), 2 * np.cos(2 * np.pi * i / 110)) if hard else (0, 0)
        M = np.float32([[1, 0, -pad - drift[0]], [0, 1, -pad - drift[1]]])
        frame = cv2.warpAffine(tissue, M, (W, H), flags=cv2.INTER_LINEAR)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        ca, sa = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        u = ((xx - cx) * ca + (yy - cy) * sa) / ax
        v = (-(xx - cx) * sa + (yy - cy) * ca) / ay
        r = np.sqrt(u * u + v * v)
        tex = cv2.remap(obj_tex, (xx - cx + 70).astype(np.float32), (yy - cy + 70).astype(np.float32), cv2.INTER_LINEAR)
        inside = r < 1.0
        rim = (r > 0.8) & (r < 1.08)
        frame = np.where(inside, 0.25 * tex, frame)                  # dark (fluid-like) interior
        frame = np.where(rim, 1.7 * tex + 0.4, frame)                # bright rim (bone-like)
        frame = frame * mask
        visible = 1
        if hard:
            if 120 <= i < 140:                                       # acoustic shadow / signal dropout
                x0 = int(cx - 45)
                shade = np.ones_like(frame)
                shade[:, max(0, x0):min(W, x0 + 90)] = 0.05
                frame = frame * shade
                visible = 0
            frame = frame + 0.18 * rng.standard_normal(frame.shape).astype(np.float32) * mask   # electronic noise (after the shadow)
        img = np.clip(frame * 110, 0, 255).astype(np.uint8)
        vw.write(img)
        bw = 2 * np.hypot(ax * ca, ay * sa)
        bh = 2 * np.hypot(ax * sa, ay * ca)
        rows.append((i, round(cx, 3), round(cy, 3), round(bw, 3), round(bh, 3), visible))
    vw.release()
    with open(a.gt, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "cx", "cy", "w", "h", "visible"])
        w.writerows(rows)
    print(f"wrote {a.out} and {a.gt}")


if __name__ == "__main__":
    main()
