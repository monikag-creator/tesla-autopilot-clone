"""
utils/metrics.py
Accumulates per-frame detection statistics and computes a summary report.
"""

import json
from collections import defaultdict
from pathlib import Path


class MetricsTracker:
    """
    Tracks inference latency and per-class detection counts across frames.

    Usage:
        tracker = MetricsTracker()
        for frame in video:
            results = model(frame)
            tracker.update(results, inference_ms)
        print(tracker.summary())
    """

    def __init__(self):
        self._latencies: list[float] = []                     # ms per frame
        self._class_counts: dict[int, int] = defaultdict(int) # total detections
        self._conf_scores: list[float] = []                   # all confidence scores

    def update(self, results, inference_ms: float) -> None:
        """
        Record one frame's results.

        Args:
            results      : Ultralytics Results object.
            inference_ms : Wall-clock inference time in milliseconds.
        """
        self._latencies.append(inference_ms)

        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                self._class_counts[cls_id] += 1
                self._conf_scores.append(conf)

    def summary(self) -> dict:
        """
        Compute and return aggregated metrics.

        Returns:
            dict with keys:
              avg_inference_ms, min_inference_ms, max_inference_ms,
              avg_fps, total_detections, avg_confidence,
              detections_per_class
        """
        if not self._latencies:
            return {"error": "No frames recorded."}

        avg_lat = sum(self._latencies) / len(self._latencies)
        avg_fps = 1000.0 / avg_lat if avg_lat > 0 else 0.0

        avg_conf = (
            sum(self._conf_scores) / len(self._conf_scores)
            if self._conf_scores else 0.0
        )

        return {
            "avg_inference_ms":   round(avg_lat, 2),
            "min_inference_ms":   round(min(self._latencies), 2),
            "max_inference_ms":   round(max(self._latencies), 2),
            "avg_fps":            round(avg_fps, 2),
            "total_detections":   sum(self._class_counts.values()),
            "avg_confidence":     round(avg_conf, 4),
            "detections_per_class": dict(self._class_counts),
        }

    def save_json(self, path: str) -> None:
        """Write the summary dict to a JSON file."""
        data = self.summary()
        Path(path).write_text(json.dumps(data, indent=2))
        print(f"[INFO] Metrics saved → {path}")
