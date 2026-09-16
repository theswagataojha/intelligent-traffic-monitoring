"""
Vehicle detection from a foreground mask using contour analysis, plus
simple feature extraction (shape, size, position) used both for filtering
false positives and for rough vehicle-type classification.

Maps to syllabus topics: Object detection, Feature extraction, Classification.
"""
import cv2
from src import config


class Detection:
    """A single detected vehicle-candidate in one frame."""

    __slots__ = ("bbox", "centroid", "area", "aspect_ratio", "vehicle_type")

    def __init__(self, bbox, area, aspect_ratio):
        x, y, w, h = bbox
        self.bbox = bbox
        self.centroid = (x + w // 2, y + h // 2)
        self.area = area
        self.aspect_ratio = aspect_ratio
        self.vehicle_type = classify_by_size(area)


def classify_by_size(area):
    """
    Very lightweight vehicle-type heuristic based on blob area.
    This is intentionally simple (an academic-prototype substitute for a
    trained classifier / YOLO model) — see README for how to swap in a
    real detector.
    """
    best_type, best_score = "unknown", None
    for vtype, (lo, hi) in config.VEHICLE_TYPE_THRESHOLDS.items():
        if lo <= area <= hi:
            # prefer the narrowest matching band
            score = hi - lo
            if best_score is None or score < best_score:
                best_type, best_score = vtype, score
    return best_type


def detect_vehicles(mask):
    """
    Find vehicle candidates in a binary foreground mask.

    Returns a list of Detection objects, already filtered by area and
    aspect ratio to reduce false positives from noise/shadows.
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < config.MIN_CONTOUR_AREA or area > config.MAX_CONTOUR_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h) if h > 0 else 0
        if not (config.MIN_ASPECT_RATIO <= aspect_ratio <= config.MAX_ASPECT_RATIO):
            continue

        detections.append(Detection((x, y, w, h), area, aspect_ratio))

    return detections
