"""
Multi-object centroid tracker: assigns a stable, unique ID to each detected
vehicle and follows it across frames using nearest-centroid association.

Maps to syllabus topics: Object tracking, Motion analysis, Optical-flow-like
displacement estimation (via centroid deltas between frames).
"""
from collections import OrderedDict
import numpy as np
from scipy.spatial import distance as dist
from src import config


class TrackedVehicle:
    """State kept for a single tracked vehicle across frames."""

    def __init__(self, object_id, centroid, bbox, vehicle_type):
        self.id = object_id
        self.centroid = centroid
        self.bbox = bbox
        self.vehicle_type = vehicle_type
        self.disappeared = 0
        self.trajectory = [centroid]          # history of centroids
        self.total_distance = 0.0             # cumulative pixel displacement

    def update(self, centroid, bbox, vehicle_type):
        step = dist.euclidean(self.centroid, centroid)
        self.total_distance += step
        self.centroid = centroid
        self.bbox = bbox
        self.vehicle_type = vehicle_type
        self.disappeared = 0
        self.trajectory.append(centroid)
        if len(self.trajectory) > config.DRAW_TRAJECTORY_LENGTH:
            self.trajectory.pop(0)

    def direction(self):
        """Rough movement direction based on the last few trajectory points."""
        if len(self.trajectory) < 2:
            return (0, 0)
        (x1, y1) = self.trajectory[0]
        (x2, y2) = self.trajectory[-1]
        return (x2 - x1, y2 - y1)

    def speed_px_per_frame(self):
        """Average pixel displacement per frame over the tracked lifetime."""
        n = max(len(self.trajectory) - 1, 1)
        return self.total_distance / n


class CentroidTracker:
    """
    Assigns and maintains unique IDs for vehicles across frames by matching
    new detections to existing tracks via minimum centroid distance
    (a simplified Hungarian-style greedy assignment).
    """

    def __init__(self, max_disappeared=config.MAX_DISAPPEARED_FRAMES,
                 max_distance=config.MAX_MATCH_DISTANCE):
        self.next_object_id = 0
        self.objects = OrderedDict()   # id -> TrackedVehicle
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def _register(self, centroid, bbox, vehicle_type):
        self.objects[self.next_object_id] = TrackedVehicle(
            self.next_object_id, centroid, bbox, vehicle_type
        )
        self.next_object_id += 1

    def _deregister(self, object_id):
        del self.objects[object_id]

    def update(self, detections):
        """
        detections: list of detector.Detection objects for the current frame.
        Returns the current dict of {id: TrackedVehicle}.
        """
        if len(detections) == 0:
            # no detections this frame -> age out existing tracks
            for object_id in list(self.objects.keys()):
                self.objects[object_id].disappeared += 1
                if self.objects[object_id].disappeared > self.max_disappeared:
                    self._deregister(object_id)
            return self.objects

        input_centroids = [d.centroid for d in detections]

        if len(self.objects) == 0:
            for det in detections:
                self._register(det.centroid, det.bbox, det.vehicle_type)
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[oid].centroid for oid in object_ids]

        D = dist.cdist(np.array(object_centroids), np.array(input_centroids))

        # Greedy matching: smallest distances first
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue
            object_id = object_ids[row]
            det = detections[col]
            self.objects[object_id].update(det.centroid, det.bbox, det.vehicle_type)
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(D.shape[0])) - used_rows
        unused_cols = set(range(D.shape[1])) - used_cols

        for row in unused_rows:
            object_id = object_ids[row]
            self.objects[object_id].disappeared += 1
            if self.objects[object_id].disappeared > self.max_disappeared:
                self._deregister(object_id)

        for col in unused_cols:
            det = detections[col]
            self._register(det.centroid, det.bbox, det.vehicle_type)

        return self.objects
