"""
Aggregates per-frame tracking results into traffic statistics: live vehicle
count, cumulative counts by type, and a traffic-density label.

Maps to syllabus topics: Motion analysis (aggregate), traffic-density
estimation from detection/tracking output.
"""
from src import config


def density_label(vehicle_count):
    if vehicle_count <= config.DENSITY_THRESHOLDS["LOW"]:
        return "LOW"
    if vehicle_count <= config.DENSITY_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    return "HIGH"


class TrafficAnalyzer:
    """
    Tracks, per frame, how many distinct vehicles have ever been seen
    (cumulative count, by type) and the current on-screen density.
    """

    def __init__(self):
        self.seen_ids = set()
        self.counts_by_type = {}
        self.history = []  # list of (frame_index, live_count, density_label)

    def update(self, frame_index, tracked_objects):
        live_count = len(tracked_objects)

        for obj_id, vehicle in tracked_objects.items():
            if obj_id not in self.seen_ids:
                self.seen_ids.add(obj_id)
                vtype = vehicle.vehicle_type
                self.counts_by_type[vtype] = self.counts_by_type.get(vtype, 0) + 1

        label = density_label(live_count)
        self.history.append((frame_index, live_count, label))
        return live_count, label

    def summary(self):
        return {
            "total_unique_vehicles": len(self.seen_ids),
            "counts_by_type": dict(self.counts_by_type),
            "frames_processed": len(self.history),
        }
