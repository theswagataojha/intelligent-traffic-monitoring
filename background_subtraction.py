"""
Background modeling and foreground extraction.

Maps to syllabus topics: Background subtraction, Segmentation, Edge detection
(used indirectly through morphological cleanup of the mask boundary).
"""
import cv2
import numpy as np
from src import config


class BackgroundSubtractor:
    """
    Wraps OpenCV's MOG2 background subtractor and adds morphological
    cleanup so the resulting foreground mask contains solid, contour-friendly
    blobs for the vehicles instead of noisy speckles.
    """

    def __init__(self):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=config.BG_HISTORY,
            varThreshold=config.BG_VAR_THRESHOLD,
            detectShadows=config.BG_DETECT_SHADOWS,
        )
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, config.MORPH_KERNEL_SIZE
        )

    def apply(self, frame):
        """
        Returns a clean binary foreground mask (uint8, 0/255) for the given
        frame. Shadows (gray value 127 in MOG2 output) are removed.
        """
        mask = self.subtractor.apply(frame)

        # Remove shadow pixels (MOG2 marks them as 127 instead of 255)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological opening removes small noise specks
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, self._kernel,
            iterations=config.MORPH_OPEN_ITERATIONS,
        )
        # Morphological closing fills small holes inside vehicle blobs
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, self._kernel,
            iterations=config.MORPH_CLOSE_ITERATIONS,
        )
        return mask

    def edges_from_mask(self, mask):
        """Optional: extract edges of foreground blobs (Canny)."""
        return cv2.Canny(mask, 50, 150)
