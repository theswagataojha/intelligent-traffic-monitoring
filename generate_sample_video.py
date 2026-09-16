#!/usr/bin/env python3
"""
Generates a small synthetic traffic video (moving rectangles of varying
sizes on a static road background) purely so the pipeline can be smoke
tested without needing to download a real dataset.

This is a TEST UTILITY, not part of the production pipeline. For real
results, run main.py on genuine traffic footage (see README).
"""
import os
import numpy as np
import cv2

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "sample_data", "synthetic_traffic.mp4")
WIDTH, HEIGHT, FPS, DURATION_SEC = 960, 540, 25, 6


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT_PATH, fourcc, FPS, (WIDTH, HEIGHT))

    n_frames = FPS * DURATION_SEC

    # a few "vehicles" of different sizes/speeds moving left->right or right->left
    vehicles = [
        {"x": -80, "y": 150, "w": 90, "h": 45, "vx": 6, "color": (60, 60, 200)},   # car
        {"x": WIDTH + 60, "y": 260, "w": 130, "h": 60, "vx": -5, "color": (60, 160, 60)},  # bus-ish
        {"x": -40, "y": 350, "w": 40, "h": 30, "vx": 8, "color": (200, 160, 40)},  # bike-ish
        {"x": -100, "y": 220, "w": 95, "h": 48, "vx": 4, "color": (150, 60, 150)}, # car
    ]

    for i in range(n_frames):
        frame = np.full((HEIGHT, WIDTH, 3), (90, 90, 90), dtype=np.uint8)  # road
        # lane markings
        for lane_y in (200, 320):
            for x in range(0, WIDTH, 40):
                cv2.line(frame, (x, lane_y), (x + 20, lane_y), (220, 220, 220), 2)

        for v in vehicles:
            v["x"] += v["vx"]
            x, y, w, h = int(v["x"]), int(v["y"]), v["w"], v["h"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), v["color"], -1)

        writer.write(frame)

    writer.release()
    print(f"Synthetic test video written to: {OUT_PATH}")


if __name__ == "__main__":
    main()
