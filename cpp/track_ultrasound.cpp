// BEN 2026-11-01
// Change aim: C++ port of track_ultrasound.py: template + Kalman + re-acquisition vs two baselines.
// File: track_ultrasound.cpp
// Research/portfolio code on simulated data. Not a medical device and not for diagnosis.
#include <opencv2/opencv.hpp>
#include <opencv2/video/tracking.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct Gt { double cx, cy, w, h; int vis; };
struct Pos { double x, y; int st; };

// BEN 2026-11-01
// Change aim: Read the ground-truth CSV (frame,cx,cy,w,h,visible).
// File: track_ultrasound.cpp
// Function: readGt
static std::vector<Gt> readGt(const std::string& path)
{
    std::vector<Gt> v;
    std::ifstream in(path);
    std::string line, t;
    std::getline(in, line);
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        Gt g;
        std::getline(ss, t, ',');
        std::getline(ss, t, ','); g.cx = std::stod(t);
        std::getline(ss, t, ','); g.cy = std::stod(t);
        std::getline(ss, t, ','); g.w = std::stod(t);
        std::getline(ss, t, ','); g.h = std::stod(t);
        std::getline(ss, t, ','); g.vis = std::stoi(t);
        v.push_back(g);
    }
    return v;
}

// BEN 2026-11-01
// Change aim: Template matcher with a sub-pixel cut offset; searches a window around a centre.
// File: track_ultrasound.cpp
// Function: Matcher
class Matcher {
public:
    void init(const cv::Mat& img, double cx, double cy, double w, double h)
    {
        tw_ = static_cast<int>(w) + 8;
        th_ = static_cast<int>(h) + 8;
        const int x = static_cast<int>(std::lround(cx - tw_ / 2.0)), y = static_cast<int>(std::lround(cy - th_ / 2.0));
        t_ = img(cv::Rect(x, y, tw_, th_)).clone();
        bx_ = cx - (x + tw_ / 2.0);
        by_ = cy - (y + th_ / 2.0);
    }
    // returns best score (or -1) and the best centre
    double search(const cv::Mat& img, double cx, double cy, int radius, cv::Point2d& best) const
    {
        const int bx = static_cast<int>(std::lround(cx - tw_ / 2.0)), by = static_cast<int>(std::lround(cy - th_ / 2.0));
        const int x0 = std::max(0, bx - radius), y0 = std::max(0, by - radius);
        const int x1 = std::min(img.cols, bx + tw_ + radius), y1 = std::min(img.rows, by + th_ + radius);
        if (x1 - x0 < tw_ || y1 - y0 < th_) return -1.0;
        cv::Mat res;
        cv::matchTemplate(img(cv::Rect(x0, y0, x1 - x0, y1 - y0)), t_, res, cv::TM_CCOEFF_NORMED);
        double mx;
        cv::Point loc;
        cv::minMaxLoc(res, nullptr, &mx, nullptr, &loc);
        best = cv::Point2d(x0 + loc.x + tw_ / 2.0 + bx_, y0 + loc.y + th_ / 2.0 + by_);
        return mx;
    }

private:
    cv::Mat t_;
    int tw_ = 0, th_ = 0;
    double bx_ = 0, by_ = 0;
};

// BEN 2026-11-01
// Change aim: Tracker 1: template + constant-velocity Kalman filter; coast on a poor match; wider search to re-acquire.
// File: track_ultrasound.cpp
// Function: KalmanTemplateTracker
class KalmanTemplateTracker {
public:
    void init(const cv::Mat& g, double cx, double cy, double w, double h)
    {
        m_.init(g, cx, cy, w, h);
        kf_ = cv::KalmanFilter(4, 2, 0, CV_32F);
        kf_.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);
        kf_.measurementMatrix = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0, 0, 1, 0, 0);
        kf_.processNoiseCov = (cv::Mat_<float>(4, 4) << 0.05f, 0, 0, 0, 0, 0.05f, 0, 0, 0, 0, 0.1f, 0, 0, 0, 0, 0.1f);
        cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar::all(1.0));
        kf_.errorCovPost = (cv::Mat_<float>(4, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 25, 0, 0, 0, 0, 25);
        kf_.statePost = (cv::Mat_<float>(4, 1) << static_cast<float>(cx), static_cast<float>(cy), 0.f, 0.f);
        tracking_ = true;
    }
    Pos update(const cv::Mat& g)
    {
        const cv::Mat p = kf_.predict();
        cv::Point2d z;
        const double score = m_.search(g, p.at<float>(0), p.at<float>(1), tracking_ ? 12 : 40, z);
        if (score > (tracking_ ? 0.5 : 0.65)) {
            kf_.correct((cv::Mat_<float>(2, 1) << static_cast<float>(z.x), static_cast<float>(z.y)));
            tracking_ = true;
        } else {
            kf_.statePre.copyTo(kf_.statePost);
            kf_.errorCovPre.copyTo(kf_.errorCovPost);
            tracking_ = false;
        }
        return {kf_.statePost.at<float>(0), kf_.statePost.at<float>(1), tracking_ ? 1 : 0};
    }

private:
    Matcher m_;
    cv::KalmanFilter kf_;
    bool tracking_ = true;
};

// BEN 2026-11-01
// Change aim: Baseline 1: best template match near the last position every frame.
// File: track_ultrasound.cpp
// Function: PlainMatcherTracker
class PlainMatcherTracker {
public:
    void init(const cv::Mat& g, double cx, double cy, double w, double h) { m_.init(g, cx, cy, w, h); c_ = cv::Point2d(cx, cy); }
    Pos update(const cv::Mat& g)
    {
        cv::Point2d z;
        if (m_.search(g, c_.x, c_.y, 12, z) >= -0.5) c_ = z;
        return {c_.x, c_.y, 1};
    }

private:
    Matcher m_;
    cv::Point2d c_;
};

// BEN 2026-11-01
// Change aim: Baseline 2: OpenCV MIL tracker.
// File: track_ultrasound.cpp
// Function: MilTracker
class MilTracker {
public:
    void init(const cv::Mat& f, double cx, double cy, double w, double h)
    {
        t_ = cv::TrackerMIL::create();
        box_ = cv::Rect(static_cast<int>(std::lround(cx - w / 2)), static_cast<int>(std::lround(cy - h / 2)),
                        static_cast<int>(std::lround(w)), static_cast<int>(std::lround(h)));
        t_->init(f, box_);
    }
    Pos update(const cv::Mat& f)
    {
        cv::Rect b;
        const bool ok = t_->update(f, b);
        if (ok) box_ = b;
        return {box_.x + box_.width / 2.0, box_.y + box_.height / 2.0, ok ? 1 : 0};
    }

private:
    cv::Ptr<cv::TrackerMIL> t_;
    cv::Rect box_;
};

// BEN 2026-11-01
// Change aim: IoU of two centre-size boxes.
// File: track_ultrasound.cpp
// Function: iou
static double iou(double ax, double ay, double aw, double ah, double bx, double by, double bw, double bh)
{
    const double x1 = std::max(ax - aw / 2, bx - bw / 2), x2 = std::min(ax + aw / 2, bx + bw / 2);
    const double y1 = std::max(ay - ah / 2, by - bh / 2), y2 = std::min(ay + ah / 2, by + bh / 2);
    const double inter = std::max(0.0, x2 - x1) * std::max(0.0, y2 - y1);
    return inter / (aw * ah + bw * bh - inter);
}

// BEN 2026-11-01
// Change aim: Run the three trackers on the video, print the metrics table and write CSV files.
// File: track_ultrasound.cpp
// Function: main
int main(int argc, char** argv)
{
    std::string video, gtPath, outDir = "results", tag = "run";
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--video") video = argv[i + 1];
        else if (k == "--gt") gtPath = argv[i + 1];
        else if (k == "--out") outDir = argv[i + 1];
        else if (k == "--tag") tag = argv[i + 1];
    }
    if (video.empty() || gtPath.empty()) {
        std::cerr << "usage: track_ultrasound --video v.avi --gt gt.csv [--out dir] [--tag name]\n";
        return 1;
    }
    const std::vector<Gt> gt = readGt(gtPath);
    cv::VideoCapture cap(video);
    if (!cap.isOpened()) { std::cerr << "cannot open video\n"; return 1; }

    KalmanTemplateTracker kt;
    PlainMatcherTracker pm;
    MilTracker mil;
    std::vector<Pos> tr[3];
    double ms[3] = {0, 0, 0};
    cv::Mat f3, g;
    int i = 0;
    typedef std::chrono::steady_clock Clock;
    while (cap.read(f3)) {
        cv::cvtColor(f3, g, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(g, g, cv::Size(0, 0), 1.0);
        if (i == 0) {
            kt.init(g, gt[0].cx, gt[0].cy, gt[0].w, gt[0].h);
            pm.init(g, gt[0].cx, gt[0].cy, gt[0].w, gt[0].h);
            mil.init(f3, gt[0].cx, gt[0].cy, gt[0].w, gt[0].h);
            for (auto& t : tr) t.push_back({gt[0].cx, gt[0].cy, 1});
        } else {
            Clock::time_point a = Clock::now();
            tr[0].push_back(kt.update(g));
            Clock::time_point b = Clock::now();
            tr[1].push_back(pm.update(g));
            Clock::time_point c = Clock::now();
            tr[2].push_back(mil.update(f3));
            Clock::time_point d = Clock::now();
            ms[0] += std::chrono::duration<double, std::milli>(b - a).count();
            ms[1] += std::chrono::duration<double, std::milli>(c - b).count();
            ms[2] += std::chrono::duration<double, std::milli>(d - c).count();
        }
        ++i;
    }

    const char* names[3] = {"Template + Kalman (this repo)", "Plain template matching", "OpenCV MIL"};
    const char* files[3] = {"kalman", "plain", "mil"};
    std::ostringstream t;
    t << std::fixed << std::setprecision(2);
    t << "| Tracker | mean error (px) | median (px) | max (px) | success IoU>=0.5 (%) | lost frames (>20 px) | "
         "mean error visible (px) | mean error hidden (px) | ms/frame |\n|---|---|---|---|---|---|---|---|---|\n";
    const size_t n = std::min(tr[0].size(), gt.size());
    for (int k = 0; k < 3; ++k) {
        std::vector<double> errs;
        double sv = 0, sh = 0, mx = 0;
        int nv = 0, nh = 0, ok = 0, lost = 0;
        for (size_t j = 1; j < n; ++j) {
            const double e = std::hypot(tr[k][j].x - gt[j].cx, tr[k][j].y - gt[j].cy);
            errs.push_back(e);
            mx = std::max(mx, e);
            if (e > 20) ++lost;
            if (iou(tr[k][j].x, tr[k][j].y, gt[j].w, gt[j].h, gt[j].cx, gt[j].cy, gt[j].w, gt[j].h) >= 0.5) ++ok;
            if (gt[j].vis) { sv += e; ++nv; } else { sh += e; ++nh; }
        }
        double mean = 0;
        for (double e : errs) mean += e;
        mean /= errs.size();
        std::vector<double> s = errs;
        std::sort(s.begin(), s.end());
        const double med = (s.size() % 2) ? s[s.size() / 2] : 0.5 * (s[s.size() / 2 - 1] + s[s.size() / 2]);
        t << "| " << names[k] << " | " << mean << " | " << med << " | " << mx << " | " << 100.0 * ok / errs.size() << " | " << lost
          << " | " << (nv ? sv / nv : 0.0) << " | " << (nh ? sh / nh : 0.0) << " | " << ms[k] / n << " |\n";
        std::ofstream o(outDir + "/cpp_track_" + files[k] + "_" + tag + ".csv");
        o << "frame,cx,cy,state\n";
        for (size_t j = 0; j < tr[k].size(); ++j) o << j << "," << tr[k][j].x << "," << tr[k][j].y << "," << tr[k][j].st << "\n";
    }
    std::cout << t.str();
    std::ofstream(outDir + "/cpp_metrics_" + tag + ".md") << t.str();
    return 0;
}
