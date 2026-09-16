# Intelligent Traffic Monitoring and Vehicle Tracking System

A computer-vision pipeline that processes traffic video, detects vehicles,
tracks them across frames with unique IDs, estimates traffic density, and
produces an annotated output video plus a vehicle-count-over-time graph.

Built as an academic project covering: image preprocessing, histogram
processing (CLAHE), background subtraction, object detection, feature
extraction, object tracking, motion analysis, and traffic-density
estimation — with a classical, dependency-light OpenCV pipeline (no GPU or
large pretrained model download required).

## What it does

```
Input Video → Frame Extraction → Preprocessing (resize, CLAHE, denoise)
  → Background Subtraction (MOG2 + morphology) → Vehicle Detection (contours)
  → Feature Extraction (size, aspect ratio, position) → Centroid Tracking
  (unique IDs, trajectories) → Motion & Traffic Analysis (density, counts)
  → Output Visualization (annotated video + graph + JSON stats)
```

Output includes:
- An annotated `.mp4` with bounding boxes, vehicle IDs, and trajectory trails
- A `vehicle_count_over_time.png` line graph
- A `traffic_stats.json` summary (total unique vehicles, counts by type)

## Project structure

```
traffic-monitor/
├── main.py                     # Entry point — runs the full pipeline
├── requirements.txt
├── src/
│   ├── config.py                # All tunable thresholds/parameters
│   ├── preprocessing.py          # Resize, CLAHE contrast, denoise
│   ├── background_subtraction.py # MOG2 + morphological cleanup
│   ├── detector.py               # Contour-based vehicle detection + classification
│   ├── tracker.py                # Centroid tracker (unique IDs, trajectories)
│   ├── traffic_analyzer.py       # Counts, density estimation
│   └── visualizer.py             # Drawing overlays + graph generation
├── tests/
│   ├── generate_sample_video.py  # Creates a synthetic test video (no dataset needed)
│   └── test_pipeline.py          # Unit tests (pytest)
├── sample_data/                  # Put your traffic videos here
└── output/                       # Annotated video, graph, and JSON stats are written here
```

## 1. Prerequisites

- Python 3.9 or newer
- pip
- (Optional) a webcam, or a traffic video file (`.mp4`, `.avi`, `.mov`, etc.)

Check your Python version:
```bash
python3 --version
```

## 2. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

## 3. Set up a virtual environment (recommended)

```bash
python3 -m venv venv

# Activate it:
# macOS / Linux:
source venv/bin/activate
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# Windows (cmd):
venv\Scripts\activate.bat
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

> **Headless machines (servers, CI, containers without a display):**
> `opencv-python` requires a display to open the live preview window.
> If you're on a headless environment, install `opencv-python-headless`
> instead (edit `requirements.txt` or run
> `pip install opencv-python-headless` after removing `opencv-python`),
> and always pass `--no-display` when running `main.py` (see below).

## 5. Get a traffic video to test with

You have three options:

**Option A — Use your own video.** Place any traffic footage (dashcam,
CCTV, drone, highway clip, etc.) into `sample_data/`, e.g.
`sample_data/my_traffic.mp4`.

**Option B — Use a public traffic video dataset.** Search for
freely-licensed traffic footage (e.g. Pexels, Pixabay, or academic
datasets such as UA-DETRAC) and download a clip into `sample_data/`.

**Option C — Generate a synthetic test video.** No download needed — this
creates a short clip with moving shapes so you can verify the pipeline
runs correctly end-to-end:
```bash
python tests/generate_sample_video.py
```
This writes `sample_data/synthetic_traffic.mp4`.

## 6. Run the pipeline

Basic run (opens a live preview window, press `q` to stop early):
```bash
python main.py --input sample_data/synthetic_traffic.mp4 --output output/result.mp4
```

Headless run (no display window — required on servers/CI):
```bash
python main.py --input sample_data/synthetic_traffic.mp4 --output output/result.mp4 --no-display
```

Use your webcam instead of a file:
```bash
python main.py --input 0 --output output/result.mp4
```

Quick test on only the first N frames (useful while tuning parameters):
```bash
python main.py --input sample_data/synthetic_traffic.mp4 --no-display --max-frames 100
```

All command-line options:
```bash
python main.py --help
```

| Flag | Default | Description |
|---|---|---|
| `--input`, `-i` | *(required)* | Path to input video, or `0` for the default webcam |
| `--output`, `-o` | `output/result.mp4` | Path to save the annotated output video |
| `--graph` | `output/vehicle_count_over_time.png` | Path to save the count-over-time graph |
| `--stats` | `output/traffic_stats.json` | Path to save final statistics as JSON |
| `--no-display` | off | Disable the live preview window |
| `--max-frames` | none | Cap the number of frames processed |

## 7. Check the results

After a run finishes you'll find, inside `output/`:
- `result.mp4` — annotated video with bounding boxes, vehicle IDs, and trajectory trails
- `vehicle_count_over_time.png` — a line graph of vehicle count per frame
- `traffic_stats.json` — total unique vehicles, counts by type, frames processed

Example console summary:
```
[DONE] Processing complete.
  Frames processed        : 150
  Total unique vehicles   : 5
  Counts by type          : {'car': 3, 'bike': 2}
  Time taken              : 4.3s
  Annotated video saved to: output/result.mp4
  Graph saved to          : output/vehicle_count_over_time.png
  Stats JSON saved to     : output/traffic_stats.json
```

## 8. Running the automated tests

```bash
pip install pytest
pytest tests/
```

## 9. Tuning for your own footage

All thresholds live in `src/config.py` — nothing else needs to change:
- `MIN_CONTOUR_AREA` / `MAX_CONTOUR_AREA` — filters out noise/oversized blobs; adjust based on your video's resolution and camera distance.
- `VEHICLE_TYPE_THRESHOLDS` — area ranges used for the size-based vehicle-type heuristic (bike/car/bus/truck). Re-tune by inspecting typical blob areas in your footage.
- `MAX_DISAPPEARED_FRAMES` / `MAX_MATCH_DISTANCE` — controls how forgiving the tracker is about brief occlusions vs. how far a vehicle can move between frames before it's treated as a new object.
- `DENSITY_THRESHOLDS` — vehicle-count cutoffs for LOW / MEDIUM / HIGH density labels.

## 10. Notes on the detection approach

Detection here uses classical background subtraction (MOG2) + contour
analysis, which works well for videos with a relatively static camera
(fixed CCTV/traffic-pole angle) and needs no model download — good for an
academic prototype and fully reproducible offline. Vehicle **type**
classification is a simple size-based heuristic, not a trained classifier.

To extend this into a stronger detector, you can swap `src/detector.py`'s
`detect_vehicles()` for a deep-learning detector (e.g. a pretrained YOLO
model via `ultralytics`) while keeping the rest of the pipeline
(`tracker.py`, `traffic_analyzer.py`, `visualizer.py`) unchanged — they
only depend on `Detection` objects exposing `bbox`, `centroid`, and
`vehicle_type`.

## 11. Troubleshooting

- **`Could not open video source`** — check the path in `--input` is correct and the file exists, or that a webcam is connected if using `0`.
- **Live window doesn't appear / crashes on a server** — you're likely on a headless machine; use `--no-display` and make sure `opencv-python-headless` is installed instead of `opencv-python`.
- **No vehicles detected on real footage** — your camera may be too close/far for the default `MIN_CONTOUR_AREA`/`MAX_CONTOUR_AREA`; adjust them in `src/config.py`.
- **Output video won't play** — try VLC, or change `OUTPUT_VIDEO_CODEC` in `src/config.py` to `"avc1"` or `"XVID"` (with a matching `.avi` output extension for XVID).

## License

MIT — see [LICENSE](LICENSE).
