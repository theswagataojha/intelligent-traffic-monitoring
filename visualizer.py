"""
Drawing helpers: bounding boxes, IDs, trajectories, and the on-screen
stats panel; plus the end-of-run vehicle-count-over-time graph.

Maps to syllabus topics: Output visualization / video processing.
"""
import cv2
import matplotlib
matplotlib.use("Agg")  # headless backend, no display needed
import matplotlib.pyplot as plt

BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)
TRAJECTORY_COLOR = (0, 165, 255)


def draw_tracked_vehicles(frame, tracked_objects):
    for obj_id, vehicle in tracked_objects.items():
        x, y, w, h = vehicle.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, 2)

        label = f"{vehicle.vehicle_type.upper()} #{obj_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x, y - th - 8), (x + tw + 4, y), BOX_COLOR, -1)
        cv2.putText(frame, label, (x + 2, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # trajectory trail
        pts = vehicle.trajectory
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], TRAJECTORY_COLOR, 2)

    return frame


def draw_stats_panel(frame, live_count, density, counts_by_type):
    h, w = frame.shape[:2]
    panel_h = 90
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    line1 = f"Vehicles on screen: {live_count}   |   Traffic Density: {density}"
    type_str = "  ".join(f"{k.capitalize()}: {v}" for k, v in counts_by_type.items())
    line2 = type_str if type_str else "No vehicles classified yet"

    cv2.putText(frame, line1, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, TEXT_COLOR, 2)
    cv2.putText(frame, line2, (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1)
    return frame


def save_count_over_time_graph(history, output_path):
    """
    history: list of (frame_index, live_count, density_label)
    Produces a PNG line chart of vehicle count over time.
    """
    if not history:
        return

    frames = [h[0] for h in history]
    counts = [h[1] for h in history]

    plt.figure(figsize=(10, 4))
    plt.plot(frames, counts, color="tab:blue", linewidth=1.5)
    plt.title("Vehicle Count Over Time")
    plt.xlabel("Frame Number")
    plt.ylabel("Vehicles on Screen")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
