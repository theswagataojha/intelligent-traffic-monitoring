# Project Statement: Intelligent Traffic Monitoring and Vehicle Tracking System

## Problem Statement

Traditional traffic monitoring relies heavily on manual observation or
expensive dedicated infrastructure (inductive loop sensors, radar units,
manned checkpoints), making continuous, city-scale monitoring difficult,
costly, and slow to deploy. Existing camera footage from CCTV and traffic
poles is often recorded but not actively analyzed, leaving a large amount
of usable visual data untapped for traffic management purposes.

There is a need for an automated, camera-based system that can process
ordinary traffic video, detect vehicles without specialized hardware,
track their movement across frames, and derive actionable metrics such as
vehicle count and traffic density — using only classical computer vision
techniques so it remains lightweight, transparent, and reproducible
without requiring GPU hardware or large pretrained models.

## Scope of the Project

**In scope:**
- Processing pre-recorded traffic video or a live webcam feed frame by frame.
- Image preprocessing (resizing, contrast enhancement, denoising) to
  normalize input footage.
- Background modeling and foreground extraction (MOG2 background
  subtraction) to isolate moving vehicles from a relatively static camera
  view (e.g., a fixed CCTV or traffic-pole angle).
- Detecting vehicle candidates via contour analysis and filtering by size
  and aspect ratio.
- A lightweight, heuristic, size-based vehicle-type classification (car,
  bike, bus, truck).
- Assigning each detected vehicle a persistent unique ID and tracking it
  across frames using a centroid-distance tracker.
- Estimating live vehicle count and a traffic-density label
  (LOW / MEDIUM / HIGH).
- Producing an annotated output video (bounding boxes, IDs, motion
  trails), a vehicle-count-over-time graph, and a JSON statistics summary.

**Out of scope (for this prototype):**
- Deep-learning-based object detection (e.g., YOLO) — the architecture is
  designed so a trained detector could be substituted in later, but one is
  not included by default, to keep the system dependency-light and fully
  reproducible offline.
- Speed estimation calibrated to real-world units (would require camera
  calibration against a known physical reference).
- Multi-camera fusion, lane-level analytics, wrong-way/anomaly detection,
  and automated report generation — noted as possible future extensions.
- Moving-camera footage (dashcams, drones) — the background-subtraction
  approach assumes a relatively fixed camera position.

## Target Users

- **Students and educators** studying computer vision / image processing,
  using this as a hands-on project connecting classical CV concepts
  (background subtraction, contour detection, tracking) to a real,
  end-to-end application.
- **Municipal traffic and transportation departments** looking for a
  low-cost way to analyze existing CCTV or traffic-camera footage without
  new sensor hardware.
- **Researchers and hobbyists** who want a simple, inspectable baseline
  traffic-analytics pipeline to extend or benchmark against more advanced
  (deep-learning-based) approaches.
- **Small-scale deployments** (a single intersection, a parking lot
  entrance, a campus road) where installing dedicated traffic-sensing
  hardware isn't cost-justified.

## High-Level Features

- **Video ingestion** from a file path or a live webcam, with adjustable
  frame limits for quick testing.
- **Preprocessing pipeline**: resizing, CLAHE-based contrast enhancement,
  and Gaussian denoising.
- **Background subtraction** with morphological cleanup to produce clean,
  contour-friendly foreground masks.
- **Vehicle detection** via contour analysis, filtered by configurable
  area and aspect-ratio thresholds.
- **Size-based vehicle-type classification** (car / bike / bus / truck),
  with all thresholds centralized and tunable in one configuration file.
- **Multi-object centroid tracking** that assigns persistent unique IDs
  and records each vehicle's trajectory and displacement across frames.
- **Traffic analytics**: live on-screen vehicle count, cumulative
  vehicle-type counts, and a LOW/MEDIUM/HIGH density label.
- **Output visualization**: an annotated video with bounding boxes,
  vehicle IDs, and motion trails; a vehicle-count-over-time line graph;
  and a JSON summary of the run.
- **Automated tests** covering detection, classification, and tracking
  logic using synthetically generated test data, so the pipeline can be
  verified without a real traffic video.
- **Configurable and extensible**: every threshold lives in a single
  `config.py`, and the detector is structured so a trained deep-learning
  model can be substituted in without touching the rest of the pipeline.
