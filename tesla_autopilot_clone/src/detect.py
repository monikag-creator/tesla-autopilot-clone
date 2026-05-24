"""
Tesla Autopilot Clone - Object Detection Pipeline
Uses YOLOv8 for real-time object detection on images and video.
Author: Monika | GUVI AIML Project
"""

import cv2
import time
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from utils.visualizer import draw_detections, draw_hud
from utils.metrics import MetricsTracker


# ── COCO classes relevant to autonomous driving ──────────────────────────────
DRIVING_CLASSES = {
    0:  "person",
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
    9:  "traffic light",
    11: "stop sign",
    12: "parking meter",
}

# Colour palette: one BGR colour per class id
CLASS_COLORS = {
    0:  (0,   255, 255),   # person       – yellow
    1:  (255, 165,   0),   # bicycle      – blue-orange
    2:  (0,   255,   0),   # car          – green
    3:  (255,   0, 255),   # motorcycle   – magenta
    5:  (0,   128, 255),   # bus          – orange
    7:  (0,    80, 255),   # truck        – red-orange
    9:  (255, 255,   0),   # traffic light– cyan
    11: (0,     0, 255),   # stop sign    – red
    12: (180,  90, 255),   # parking meter– purple
}


def load_model(model_path: str = "yolov8n.pt") -> YOLO:
    """Load a YOLOv8 model (downloads weights automatically on first run)."""
    print(f"[INFO] Loading model: {model_path}")
    model = YOLO(model_path)
    return model


def run_on_image(model: YOLO, image_path: str, conf: float, output_dir: str) -> dict:
    """
    Run detection on a single image.

    Returns:
        dict with keys: path, detections, inference_ms
    """
    image_path = Path(image_path)
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    t0 = time.perf_counter()
    results = model(frame, conf=conf, verbose=False)[0]
    inference_ms = (time.perf_counter() - t0) * 1000

    annotated = draw_detections(frame.copy(), results, DRIVING_CLASSES, CLASS_COLORS)
    annotated = draw_hud(annotated, inference_ms, len(results.boxes))

    out_path = Path(output_dir) / f"detected_{image_path.name}"
    cv2.imwrite(str(out_path), annotated)
    print(f"[INFO] Saved annotated image → {out_path}")

    return {
        "path": str(out_path),
        "detections": len(results.boxes),
        "inference_ms": inference_ms,
    }


def run_on_video(
    model: YOLO,
    video_source,
    conf: float,
    output_dir: str,
    save_video: bool = True,
) -> dict:
    """
    Run detection on a video file or webcam stream.

    Args:
        video_source: file path (str) or webcam index (int).
    """
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise IOError(f"Cannot open video source: {video_source}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if save_video:
        out_name = "detected_output.mp4" if isinstance(video_source, int) \
                   else f"detected_{Path(str(video_source)).name}"
        out_path = str(Path(output_dir) / out_name)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps_in, (width, height))

    tracker = MetricsTracker()
    frame_count = 0

    print("[INFO] Starting detection… Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        results = model(frame, conf=conf, verbose=False)[0]
        inference_ms = (time.perf_counter() - t0) * 1000

        annotated = draw_detections(frame.copy(), results, DRIVING_CLASSES, CLASS_COLORS)
        annotated = draw_hud(annotated, inference_ms, len(results.boxes))

        tracker.update(results, inference_ms)
        frame_count += 1

        if writer:
            writer.write(annotated)

        cv2.imshow("Tesla Autopilot Clone – Object Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
        print(f"[INFO] Saved annotated video → {out_path}")
    cv2.destroyAllWindows()

    summary = tracker.summary()
    summary["frames_processed"] = frame_count
    return summary


# ── CLI entry point ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Tesla Autopilot Clone – YOLOv8 Object Detection"
    )
    parser.add_argument("--source",  type=str,  default="0",
                        help="Image path, video path, or webcam index (default: 0)")
    parser.add_argument("--model",   type=str,  default="yolov8n.pt",
                        help="YOLOv8 model weights (yolov8n/s/m/l/x.pt)")
    parser.add_argument("--conf",    type=float, default=0.40,
                        help="Confidence threshold (0–1, default: 0.40)")
    parser.add_argument("--output",  type=str,  default="output",
                        help="Output directory for results")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save the annotated video/image")
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)

    model = load_model(args.model)

    # Determine source type
    source = args.source
    if source.isdigit():
        source = int(source)          # webcam index

    if isinstance(source, int):
        summary = run_on_video(model, source, args.conf, args.output,
                               save_video=not args.no_save)
        print("\n=== Detection Summary ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    elif Path(source).suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        summary = run_on_video(model, source, args.conf, args.output,
                               save_video=not args.no_save)
        print("\n=== Detection Summary ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    else:
        info = run_on_image(model, source, args.conf, args.output)
        print(f"\n[RESULT] Detected {info['detections']} objects "
              f"in {info['inference_ms']:.1f} ms → {info['path']}")


if __name__ == "__main__":
    main()
