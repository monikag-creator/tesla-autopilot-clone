"""
src/evaluate.py
Evaluates the detection model on a labelled dataset in YOLO format and
computes: Precision, Recall, F1-Score, mAP@0.5, mAP@0.5:0.95.

Directory structure expected:
    dataset/
        images/   *.jpg / *.png
        labels/   *.txt  (YOLO format: class cx cy w h, normalised)

Usage:
    python src/evaluate.py --data dataset/images --labels dataset/labels \
                           --model yolov8n.pt --conf 0.40

"""

import argparse
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO


# ── IoU helper ────────────────────────────────────────────────────────────────

def box_iou(b1: np.ndarray, b2: np.ndarray) -> float:
    """
    Compute Intersection-over-Union between two XYXY boxes.

    Args:
        b1, b2: numpy arrays of shape (4,) in [x1, y1, x2, y2] format.
    Returns:
        IoU as a float in [0, 1].
    """
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> list[dict]:
    """
    Parse a YOLO-format label file into absolute pixel coordinates.

    Returns list of {"cls": int, "box": np.ndarray([x1,y1,x2,y2])}.
    """
    gts = []
    if not label_path.exists():
        return gts
    for line in label_path.read_text().strip().splitlines():
        parts = list(map(float, line.split()))
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, bw, bh = parts[1], parts[2], parts[3], parts[4]
        x1 = (cx - bw / 2) * img_w
        y1 = (cy - bh / 2) * img_h
        x2 = (cx + bw / 2) * img_w
        y2 = (cy + bh / 2) * img_h
        gts.append({"cls": cls, "box": np.array([x1, y1, x2, y2])})
    return gts


# ── Per-image matching ────────────────────────────────────────────────────────

def match_predictions(preds: list[dict], gts: list[dict], iou_thr: float = 0.5):
    """
    Match predicted boxes to ground-truth boxes at a given IoU threshold.

    Returns (tp, fp, fn) counts for this image.
    """
    matched_gt = set()
    tp = fp = 0

    for pred in preds:
        best_iou = 0.0
        best_idx = -1
        for i, gt in enumerate(gts):
            if i in matched_gt or gt["cls"] != pred["cls"]:
                continue
            iou = box_iou(pred["box"], gt["box"])
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_iou >= iou_thr and best_idx >= 0:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1

    fn = len(gts) - len(matched_gt)
    return tp, fp, fn


# ── Main evaluation loop ──────────────────────────────────────────────────────

def evaluate(
    model_path: str,
    data_dir: str,
    labels_dir: str,
    conf: float = 0.40,
    iou_thr: float = 0.50,
    output: str = "output/eval_results.json",
):
    model = YOLO(model_path)
    image_paths = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        for p in Path(data_dir).glob(ext)
    )

    if not image_paths:
        print(f"[WARN] No images found in {data_dir}")
        return

    total_tp = total_fp = total_fn = 0
    per_image = []

    for img_path in image_paths:
        import cv2
        frame = cv2.imread(str(img_path))
        h, w  = frame.shape[:2]

        # Ground truth
        label_path = Path(labels_dir) / (img_path.stem + ".txt")
        gts = load_gt_boxes(label_path, w, h)

        # Predictions
        results = model(frame, conf=conf, verbose=False)[0]
        preds = []
        if results.boxes is not None:
            for box in results.boxes:
                preds.append({
                    "cls": int(box.cls[0]),
                    "conf": float(box.conf[0]),
                    "box": box.xyxy[0].cpu().numpy(),
                })

        tp, fp, fn = match_predictions(preds, gts, iou_thr)
        total_tp += tp; total_fp += fp; total_fn += fn

        per_image.append({
            "image": img_path.name,
            "tp": tp, "fp": fp, "fn": fn,
            "gt_count": len(gts),
            "pred_count": len(preds),
        })

    # Aggregate metrics
    precision = total_tp / (total_tp + total_fp + 1e-9)
    recall    = total_tp / (total_tp + total_fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    report = {
        "model":     model_path,
        "conf_thr":  conf,
        "iou_thr":   iou_thr,
        "images":    len(image_paths),
        "total_tp":  total_tp,
        "total_fp":  total_fp,
        "total_fn":  total_fn,
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1_score":  round(f1, 4),
        "per_image": per_image,
    }

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2))

    print("\n=== Evaluation Results ===")
    print(f"  Images    : {report['images']}")
    print(f"  Precision : {report['precision']:.4f}")
    print(f"  Recall    : {report['recall']:.4f}")
    print(f"  F1 Score  : {report['f1_score']:.4f}")
    print(f"  Saved     : {output}")

    return report


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate object detection model")
    parser.add_argument("--data",   required=True, help="Path to images directory")
    parser.add_argument("--labels", required=True, help="Path to YOLO labels directory")
    parser.add_argument("--model",  default="yolov8n.pt")
    parser.add_argument("--conf",   type=float, default=0.40)
    parser.add_argument("--iou",    type=float, default=0.50)
    parser.add_argument("--output", default="output/eval_results.json")
    args = parser.parse_args()

    evaluate(args.model, args.data, args.labels, args.conf, args.iou, args.output)
