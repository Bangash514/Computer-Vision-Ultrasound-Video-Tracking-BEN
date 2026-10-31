# BEN 2026-11-01
# Change aim: Track a structure in an ultrasound-like video: template matching + Kalman filter + re-acquisition.
# File: track_ultrasound.py
"""Compares three trackers on a video with ground truth: (1) template + Kalman + gating (this repo),
(2) plain template matching (best match every frame), (3) OpenCV MIL.
Research/portfolio code on simulated data. Not a medical device and not for diagnosis."""
import argparse
import csv
import time
import numpy as np
import cv2


# BEN 2026-11-01
# Change aim: Read the ground-truth CSV.
# File: track_ultrasound.py
# Function: read_gt
def read_gt(path):
    return np.genfromtxt(path, delimiter=",", skip_header=1)


# BEN 2026-11-01
# Change aim: Template matcher: cut a template at the start, search a window around a centre, return best centre and score.
# File: track_ultrasound.py
# Function: Matcher
class Matcher:
    def __init__(self, img, cx, cy, w, h):
        self.tw, self.th = int(w) + 8, int(h) + 8
        x, y = int(round(cx - self.tw / 2)), int(round(cy - self.th / 2))
        self.t = img[y:y + self.th, x:x + self.tw].copy()
        self.bx, self.by = cx - (x + self.tw / 2), cy - (y + self.th / 2)

    def search(self, img, c, radius):
        x0 = max(0, int(round(c[0] - self.tw / 2)) - radius)
        y0 = max(0, int(round(c[1] - self.th / 2)) - radius)
        x1 = min(img.shape[1], int(round(c[0] - self.tw / 2)) + self.tw + radius)
        y1 = min(img.shape[0], int(round(c[1] - self.th / 2)) + self.th + radius)
        if x1 - x0 < self.tw or y1 - y0 < self.th:
            return None, -1.0
        res = cv2.matchTemplate(img[y0:y1, x0:x1], self.t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        return (x0 + loc[0] + self.tw / 2 + self.bx, y0 + loc[1] + self.th / 2 + self.by), mx


# BEN 2026-11-01
# Change aim: Tracker 1: template + constant-velocity Kalman filter; coast when the match is poor; wider search to re-acquire.
# File: track_ultrasound.py
# Function: KalmanTemplateTracker
class KalmanTemplateTracker:
    def __init__(self, img, cx, cy, w, h):
        self.m = Matcher(img, cx, cy, w, h)
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.processNoiseCov = np.diag([0.05, 0.05, 0.1, 0.1]).astype(np.float32)
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        kf.errorCovPost = np.diag([1, 1, 25, 25]).astype(np.float32)
        kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
        self.kf, self.tracking = kf, True

    def update(self, img):
        p = self.kf.predict()
        pc = (float(p[0, 0]), float(p[1, 0]))
        z, score = self.m.search(img, pc, 12 if self.tracking else 40)
        need = 0.5 if self.tracking else 0.65
        if z is not None and score > need:
            self.kf.correct(np.array([[z[0]], [z[1]]], np.float32))
            self.tracking = True
        else:
            self.kf.statePost = self.kf.statePre.copy()
            self.kf.errorCovPost = self.kf.errorCovPre.copy()
            self.tracking = False
        s = self.kf.statePost
        return float(s[0, 0]), float(s[1, 0]), int(self.tracking)


# BEN 2026-11-01
# Change aim: Baseline 1: take the best template match near the last position every frame, no filter, no gating.
# File: track_ultrasound.py
# Function: PlainMatcherTracker
class PlainMatcherTracker:
    def __init__(self, img, cx, cy, w, h):
        self.m, self.c = Matcher(img, cx, cy, w, h), (cx, cy)

    def update(self, img):
        z, _ = self.m.search(img, self.c, 12)
        if z is not None:
            self.c = z
        return self.c[0], self.c[1], 1


# BEN 2026-11-01
# Change aim: Baseline 2: OpenCV MIL tracker.
# File: track_ultrasound.py
# Function: MilTracker
class MilTracker:
    def __init__(self, img3, cx, cy, w, h):
        self.t = cv2.TrackerMIL_create()
        self.box = (cx - w / 2, cy - h / 2, w, h)
        self.t.init(img3, tuple(int(round(v)) for v in self.box))

    def update(self, img3):
        ok, b = self.t.update(img3)
        if ok:
            self.box = b
        return self.box[0] + self.box[2] / 2, self.box[1] + self.box[3] / 2, int(ok)


# BEN 2026-11-01
# Change aim: IoU of two centre-size boxes.
# File: track_ultrasound.py
# Function: iou
def iou(a, b):
    x1, x2 = max(a[0] - a[2] / 2, b[0] - b[2] / 2), min(a[0] + a[2] / 2, b[0] + b[2] / 2)
    y1, y2 = max(a[1] - a[3] / 2, b[1] - b[3] / 2), min(a[1] + a[3] / 2, b[1] + b[3] / 2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


# BEN 2026-11-01
# Change aim: Run all three trackers, print metrics, write CSV files.
# File: track_ultrasound.py
# Function: main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()
    gt = read_gt(a.gt)
    cap = cv2.VideoCapture(a.video)
    names = ["Template + Kalman (this repo)", "Plain template matching", "OpenCV MIL"]
    trackers, tracks, ms = None, [[], [], []], [0.0, 0.0, 0.0]
    i = 0
    while True:
        ok, f3 = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(f3, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (0, 0), 1.0)
        if i == 0:
            args = (gt[0, 1], gt[0, 2], gt[0, 3], gt[0, 4])
            trackers = [KalmanTemplateTracker(g, *args), PlainMatcherTracker(g, *args), MilTracker(f3, *args)]
            for t in tracks:
                t.append((gt[0, 1], gt[0, 2], 1))
        else:
            for k, tr in enumerate(trackers):
                t0 = time.perf_counter()
                tracks[k].append(tr.update(f3 if k == 2 else g))
                ms[k] += (time.perf_counter() - t0) * 1000
        i += 1
    lines = ["| Tracker | mean error (px) | median (px) | max (px) | success IoU>=0.5 (%) | lost frames (>20 px) | "
             "mean error visible (px) | mean error hidden (px) | ms/frame |", "|---|---|---|---|---|---|---|---|---|"]
    for k, name in enumerate(names):
        tr = np.array(tracks[k])
        n = min(len(tr), len(gt))
        err = np.hypot(tr[1:n, 0] - gt[1:n, 1], tr[1:n, 1] - gt[1:n, 2])
        ious = [iou((tr[j, 0], tr[j, 1], gt[j, 3], gt[j, 4]), (gt[j, 1], gt[j, 2], gt[j, 3], gt[j, 4])) for j in range(1, n)]
        vis = gt[1:n, 5] == 1
        hid = ~vis
        lines.append(f"| {name} | {err.mean():.2f} | {np.median(err):.2f} | {err.max():.2f} | {100 * np.mean(np.array(ious) >= 0.5):.2f} | "
                     f"{int((err > 20).sum())} | {err[vis].mean():.2f} | {(err[hid].mean() if hid.any() else 0):.2f} | {ms[k] / n:.2f} |")
        with open(f"{a.out}/py_track_{['kalman', 'plain', 'mil'][k]}_{a.tag}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "cx", "cy", "state"])
            for j, r in enumerate(tr):
                w.writerow([j, r[0], r[1], int(r[2])])
    text = "\n".join(lines)
    print(text)
    open(f"{a.out}/py_metrics_{a.tag}.md", "w").write(text + "\n")


if __name__ == "__main__":
    main()
