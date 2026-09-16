# Project Report: Intelligent Traffic Monitoring and Vehicle Tracking Using Computer Vision

## 1. Problem Statement

Traditional traffic monitoring systems rely heavily on manual observation or
expensive dedicated infrastructure, making continuous monitoring difficult
and inefficient. This project implements an automated computer-vision
system that analyzes traffic video, detects vehicles, tracks their
movement, and extracts useful information such as vehicle count, movement
patterns, and traffic density.

## 2. Objectives

- Process and enhance traffic video frames using image-processing techniques.
- Detect vehicles automatically from video sequences.
- Track detected vehicles across consecutive frames.
- Analyze vehicle movement and motion patterns.
- Estimate traffic density based on the number of detected vehicles.
- Visualize vehicle trajectories and tracking results.

## 3. Concept-to-implementation mapping

| Concept | Implementation | File |
|---|---|---|
| Image preprocessing | Resize, Gaussian blur denoising | `src/preprocessing.py` |
| Histogram processing | CLAHE contrast enhancement | `src/preprocessing.py` |
| Background subtraction | MOG2 background model + morphology | `src/background_subtraction.py` |
| Segmentation | Foreground mask cleanup (open/close) | `src/background_subtraction.py` |
| Object detection | Contour-based blob detection | `src/detector.py` |
| Feature extraction | Area, aspect ratio, position, size-based type | `src/detector.py` |
| Classification | Size-based vehicle-type heuristic | `src/detector.py` |
| Object tracking | Centroid tracker with unique IDs | `src/tracker.py` |
| Motion analysis | Per-track displacement, direction, speed | `src/tracker.py` |
| Traffic analysis | Live count, cumulative counts, density label | `src/traffic_analyzer.py` |
| Output visualization | Bounding boxes, IDs, trails, stats panel, graph | `src/visualizer.py` |

## 4. Methodology

```
Input Traffic Video
        ↓
Frame Extraction
        ↓
Image Preprocessing (resize, CLAHE, denoise)
        ↓
Background / Foreground Separation (MOG2 + morphology)
        ↓
Vehicle Detection (contour analysis)
        ↓
Feature Extraction (shape, size, position)
        ↓
Vehicle Tracking (centroid matching, unique IDs)
        ↓
Motion Analysis (direction, displacement, speed)
        ↓
Traffic Analysis (vehicle count, density, type distribution)
        ↓
Output Visualization (annotated video, graph, stats)
```

## 5. Technologies used

- **Language:** Python 3
- **Libraries:** OpenCV, NumPy, SciPy (for tracker distance computation), Matplotlib

## 6. Scope and possible extensions

Current scope: vehicle detection, tracking, counting, and density
estimation on general traffic footage (cars, bikes, buses, trucks —
distinguished by a size heuristic).

Possible extensions (not implemented here, but the architecture supports
them):
- Swapping the contour-based detector for a trained deep-learning detector
  (e.g. YOLO) for more accurate detection/classification.
- Speed estimation calibrated against real-world distances (needs camera
  calibration / a known reference scale).
- Wrong-direction detection, lane-wise counting, congestion/anomaly
  detection, and automatic traffic reports.

## 7. Expected outcomes

By running `main.py` on traffic footage, the system produces:
- Vehicles detected and tracked with persistent unique IDs across frames.
- Vehicle-type distribution (heuristic-based).
- A traffic-density label (LOW / MEDIUM / HIGH) per frame.
- An annotated output video and a vehicle-count-over-time graph.
