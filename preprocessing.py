"""
Frame preprocessing: resize, denoise, contrast-enhance.

Maps to syllabus topics: Image preprocessing, Histogram processing.
"""
import cv2
from src import config


def resize_frame(frame, width=config.RESIZE_WIDTH):
    """Resize a frame to a fixed width, preserving aspect ratio."""
    h, w = frame.shape[:2]
    if w == width:
        return frame
    scale = width / float(w)
    new_dim = (width, int(h * scale))
    return cv2.resize(frame, new_dim, interpolation=cv2.INTER_AREA)


def denoise(frame):
    """Reduce sensor/compression noise with a Gaussian blur."""
    return cv2.GaussianBlur(frame, config.GAUSSIAN_BLUR_KERNEL, 0)


def enhance_contrast(frame):
    """
    Improve poorly illuminated frames using CLAHE (Contrast Limited
    Adaptive Histogram Equalization) applied to the luminance channel.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_GRID_SIZE,
    )
    l_enhanced = clahe.apply(l_channel)

    merged = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def preprocess_frame(frame):
    """Full preprocessing pipeline applied to every incoming frame."""
    frame = resize_frame(frame)
    frame = enhance_contrast(frame)
    frame = denoise(frame)
    return frame
