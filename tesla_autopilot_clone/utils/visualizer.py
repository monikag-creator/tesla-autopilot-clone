"""
utils/visualizer.py
Annotation helpers for drawing bounding boxes, labels, and HUD overlay.
"""

import cv2
import numpy as np


def draw_detections(frame: np.ndarray, results, class_map: dict, color_map: dict) -> np.ndarray:
    """
    Draw bounding boxes and class labels for detections that belong to
    the autonomous-driving class subset.

    Args:
        frame      : BGR image (will be modified in-place and returned).
        results    : Ultralytics Results object (from model inference).
        class_map  : {class_id: label_string} – only these IDs are drawn.
        color_map  : {class_id: (B, G, R)} BGR colour per class.

    Returns:
        Annotated frame.
    """
    if results.boxes is None:
        return frame

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in class_map:
            continue

        conf  = float(box.conf[0])
        label = f"{class_map[cls_id]} {conf:.2f}"
        color = color_map.get(cls_id, (255, 255, 255))

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # ── Bounding box ──────────────────────────────────────────────────
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)

        # ── Label background pill ─────────────────────────────────────────
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness  = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        pad = 3

        label_y1 = max(y1 - th - 2 * pad, 0)
        label_y2 = label_y1 + th + 2 * pad
        label_x2 = x1 + tw + 2 * pad

        cv2.rectangle(frame, (x1, label_y1), (label_x2, label_y2), color, -1)

        # Decide text colour for readability
        brightness = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
        text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)

        cv2.putText(
            frame, label,
            (x1 + pad, label_y2 - pad - baseline),
            font, font_scale, text_color, thickness, cv2.LINE_AA
        )

    return frame


def draw_hud(frame: np.ndarray, inference_ms: float, n_objects: int) -> np.ndarray:
    """
    Overlay a Heads-Up Display in the top-left corner showing:
      • Inference time (ms)
      • FPS equivalent
      • Object count

    Args:
        frame        : BGR image.
        inference_ms : Inference latency in milliseconds.
        n_objects    : Number of detections in this frame.

    Returns:
        Frame with HUD overlay.
    """
    fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0
    lines = [
        f"Inference : {inference_ms:6.1f} ms",
        f"FPS       : {fps:6.1f}",
        f"Objects   : {n_objects:6d}",
    ]

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.50
    thickness  = 1
    pad        = 6
    line_h     = 20

    # Compute HUD background size
    max_w = max(cv2.getTextSize(l, font, font_scale, thickness)[0][0] for l in lines)
    bg_x2 = 2 * pad + max_w
    bg_y2 = 2 * pad + len(lines) * line_h

    # Semi-transparent dark rectangle
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (bg_x2, bg_y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, line in enumerate(lines):
        y = pad + (i + 1) * line_h - 4
        cv2.putText(frame, line, (pad, y), font, font_scale,
                    (0, 220, 100), thickness, cv2.LINE_AA)

    return frame
