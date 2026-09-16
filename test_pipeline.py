"""
Lightweight sanity tests for the core pipeline components.
Run with:  pytest tests/
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.detector import detect_vehicles, classify_by_size
from src.tracker import CentroidTracker
from src.traffic_analyzer import density_label


def make_blob_mask(boxes, shape=(300, 400)):
    mask = np.zeros(shape, dtype=np.uint8)
    for (x, y, w, h) in boxes:
        mask[y:y + h, x:x + w] = 255
    return mask


def test_detect_vehicles_finds_expected_blobs():
    mask = make_blob_mask([(50, 50, 60, 40), (200, 100, 80, 50)])
    detections = detect_vehicles(mask)
    assert len(detections) == 2


def test_classify_by_size_returns_known_label():
    label = classify_by_size(5000)
    assert label in ("bike", "car", "bus", "truck", "unknown")


def test_tracker_assigns_stable_ids_across_frames():
    tracker = CentroidTracker()

    mask1 = make_blob_mask([(10, 10, 50, 30)])
    detections1 = detect_vehicles(mask1)
    objects1 = tracker.update(detections1)
    assert len(objects1) == 1
    first_id = list(objects1.keys())[0]

    # same vehicle moved slightly to the right -> should keep the same ID
    mask2 = make_blob_mask([(15, 10, 50, 30)])
    detections2 = detect_vehicles(mask2)
    objects2 = tracker.update(detections2)
    assert len(objects2) == 1
    assert list(objects2.keys())[0] == first_id


def test_density_label_thresholds():
    assert density_label(1) == "LOW"
    assert density_label(8) == "MEDIUM"
    assert density_label(20) == "HIGH"
