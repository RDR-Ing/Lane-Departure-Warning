# Lane Departure Warning System (LDW)

This project implements a basic Lane Departure Warning (LDW) system using
classical computer vision techniques in OpenCV.

## Features
- Lane detection using Canny + Hough Transform
- Region of Interest (ROI) filtering
- Lane smoothing using exponential moving average
- Robust lane tracking against vehicle interference
- Ego-lane based departure detection

## Tech Stack
- Python
- OpenCV
- NumPy

## Run
```bash
python lane_ldw.py --video path\to\video.mp4 --scale 0.6
