#!/usr/bin/env python3
"""
Intelligent Traffic Monitoring and Vehicle Tracking System
------------------------------------------------------------
Entry point. Runs the full pipeline described in the project report:

    Input video -> Frame extraction -> Preprocessing ->
    Background subtraction -> Vehicle detection -> Feature extraction ->
    Tracking -> Motion analysis -> Traffic analysis -> Output visualization

Usage:
    python main.py --input sample_data/traffic.mp4 --output output/result.mp4

Run `python main.py --help` for all options.
"""
import argparse
import json
import os
import sys
import time

import cv2

from src import config
from src.preprocessing import preprocess_frame
from src.background_subtraction import BackgroundSubtractor
from src.detector import detect_vehicles
from src.tracker import CentroidTracker
from src.traffic_analyzer import TrafficAnalyzer
from src.visualizer import draw_tracked_vehicles, draw_stats_panel, save_count_over_time_graph


def parse_args():
    parser = argparse.ArgumentParser(
        description="Intelligent Traffic Monitoring and Vehicle Tracking"
    )
    parser.add_argument("--input", "-i", required=True,
                         help="Path to input traffic video, or 0 for webcam")
    parser.add_argument("--output", "-o", default="output/result.mp4",
                         help="Path to save the annotated output video")
    parser.add_argument("--graph", default="output/vehicle_count_over_time.png",
                         help="Path to save the vehicle-count-over-time graph")
    parser.add_argument("--stats", default="output/traffic_stats.json",
                         help="Path to save final traffic statistics as JSON")
    parser.add_argument("--no-display", action="store_true",
                         help="Disable the live preview window (use in headless environments)")
    parser.add_argument("--max-frames", type=int, default=None,
                         help="Optional cap on number of frames to process (useful for quick tests)")
    return parser.parse_args()


def open_video_source(source):
    # allow "0" to mean the default webcam
    if source == "0":
        source = 0
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"Could not open video source: {source}")
    return cap


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.graph) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.stats) or ".", exist_ok=True)

    cap = open_video_source(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    bg_subtractor = BackgroundSubtractor()
    tracker = CentroidTracker()
    analyzer = TrafficAnalyzer()

    writer = None
    frame_index = 0
    start_time = time.time()

    print(f"[INFO] Starting processing. Source FPS: {fps:.2f}, "
          f"total frames reported: {total_frames if total_frames > 0 else 'unknown'}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_index += 1
            if args.max_frames and frame_index > args.max_frames:
                break

            # 1. Preprocessing
            frame = preprocess_frame(frame)

            # 2. Background subtraction -> foreground mask
            mask = bg_subtractor.apply(frame)

            # 3. Vehicle detection + feature extraction (in detector.py)
            detections = detect_vehicles(mask)

            # 4. Tracking (assigns/maintains unique IDs, motion analysis)
            tracked_objects = tracker.update(detections)

            # 5. Traffic analysis (counts, density)
            live_count, density = analyzer.update(frame_index, tracked_objects)

            # 6. Output visualization
            annotated = draw_tracked_vehicles(frame.copy(), tracked_objects)
            annotated = draw_stats_panel(
                annotated, live_count, density, analyzer.counts_by_type
            )

            if writer is None:
                h, w = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*config.OUTPUT_VIDEO_CODEC)
                writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

            writer.write(annotated)

            if not args.no_display:
                cv2.imshow("Intelligent Traffic Monitoring", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Interrupted by user (q pressed).")
                    break

            if frame_index % 50 == 0:
                elapsed = time.time() - start_time
                print(f"[INFO] Processed {frame_index} frames "
                      f"({frame_index / elapsed:.1f} fps) | "
                      f"live count: {live_count} | density: {density}")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    # Final outputs
    save_count_over_time_graph(analyzer.history, args.graph)

    summary = analyzer.summary()
    summary["annotated_video"] = args.output
    summary["count_over_time_graph"] = args.graph
    with open(args.stats, "w") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - start_time
    print("\n[DONE] Processing complete.")
    print(f"  Frames processed        : {summary['frames_processed']}")
    print(f"  Total unique vehicles   : {summary['total_unique_vehicles']}")
    print(f"  Counts by type          : {summary['counts_by_type']}")
    print(f"  Time taken              : {elapsed:.1f}s")
    print(f"  Annotated video saved to: {args.output}")
    print(f"  Graph saved to          : {args.graph}")
    print(f"  Stats JSON saved to     : {args.stats}")


if __name__ == "__main__":
    sys.exit(main())
