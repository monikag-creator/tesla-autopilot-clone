"""
src/train.py
Fine-tune YOLOv8 on a custom driving dataset.

Dataset must follow Ultralytics YOLO format:
    dataset/
        images/train/    images/val/
        labels/train/    labels/val/
        data.yaml

data.yaml example:
    path: /abs/path/to/dataset
    train: images/train
    val:   images/val
    nc: 9
    names: [person, bicycle, car, motorcycle, bus, truck,
            traffic light, stop sign, parking meter]

Usage:
    python src/train.py --data dataset/data.yaml --model yolov8n.pt \
                        --epochs 50 --batch 16 --name autopilot_v1
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train(
    data_yaml: str,
    base_model: str = "yolov8n.pt",
    epochs: int = 50,
    img_size: int = 640,
    batch: int = 16,
    lr0: float = 0.01,
    run_name: str = "autopilot_clone",
    project: str = "runs/train",
    device: str = "",          # "" = auto-select GPU/CPU
):
    """
    Fine-tune a YOLOv8 model on the supplied dataset.

    Args:
        data_yaml  : Path to dataset YAML configuration file.
        base_model : Pretrained weights to start from.
        epochs     : Number of training epochs.
        img_size   : Input resolution (pixels, square).
        batch      : Batch size (use -1 for auto).
        lr0        : Initial learning rate.
        run_name   : Name for this training run.
        project    : Root directory for training artefacts.
        device     : Torch device string, e.g. "0", "cpu", or "" (auto).
    """
    print(f"[INFO] Starting fine-tune: {base_model} → {run_name}")
    print(f"       Dataset  : {data_yaml}")
    print(f"       Epochs   : {epochs} | Img size: {img_size} | Batch: {batch}")

    model = YOLO(base_model)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        lr0=lr0,
        name=run_name,
        project=project,
        device=device,
        # Augmentation — helps generalise on limited data
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        # Save best checkpoint
        save=True,
        save_period=10,
        val=True,
    )

    best_weights = Path(project) / run_name / "weights" / "best.pt"
    print(f"\n[INFO] Training complete.")
    print(f"       Best weights → {best_weights}")
    return str(best_weights)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 for autopilot clone")
    parser.add_argument("--data",    required=True,  help="data.yaml path")
    parser.add_argument("--model",   default="yolov8n.pt")
    parser.add_argument("--epochs",  type=int,   default=50)
    parser.add_argument("--batch",   type=int,   default=16)
    parser.add_argument("--imgsz",   type=int,   default=640)
    parser.add_argument("--lr0",     type=float, default=0.01)
    parser.add_argument("--name",    default="autopilot_clone")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--device",  default="",
                        help="Device: '0', 'cpu', etc. (default: auto)")
    args = parser.parse_args()

    train(
        args.data, args.model, args.epochs,
        args.imgsz, args.batch, args.lr0,
        args.name, args.project, args.device,
    )
