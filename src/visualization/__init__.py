# src/visualization/__init__.py
"""
MODULE VISUALIZATION
- [TV6] display_segmentation.py, save_results.py
- [TV4] display_detection.py
"""
# [TV6] - Segmentation
from .display_segmentation import (
    overlay_mask,
    draw_contour,
    make_side_by_side,
    visualize_segmentation,
)
from .save_results import save_all_results

# [TV4] - Detection
from .display_detection import (
    draw_boxes,
    draw_single_box,
    make_detection_panel,
    visualize_detection,
    filter_by_score,
    get_color,
    COLORS,
)

__all__ = [
    # Segmentation (TV6)
    "overlay_mask", "draw_contour", "make_side_by_side",
    "visualize_segmentation", "save_all_results",
    # Detection (TV4)
    "draw_boxes", "draw_single_box", "make_detection_panel",
    "visualize_detection", "filter_by_score", "get_color", "COLORS",
]
