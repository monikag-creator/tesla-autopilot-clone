# 🚗 Tesla Autopilot Clone — Object Detection

**GUVI AIML Course Project | Monika**

A Python-based autonomous driving object detection system inspired by Tesla Autopilot, built using **YOLOv8** (You Only Look Once v8) for real-time detection of road objects such as cars, pedestrians, traffic lights, trucks, and stop signs.

---

## 📌 Project Overview

This project implements an end-to-end object detection pipeline that:

- Loads and preprocesses video frames or image datasets
- Runs inference using a pre-trained / fine-tuned **YOLOv8** model
- Draws annotated bounding boxes with class labels and confidence scores
- Displays a real-time HUD showing inference speed and object count
- Evaluates performance with Precision, Recall, F1-Score, and IoU metrics
- Supports training / fine-tuning on custom driving datasets

---

## 🧠 Model Selection — Why YOLOv8?

| Model | Speed | Accuracy | Notes |
|-------|-------|----------|-------|
| YOLOv8n | ⚡⚡⚡⚡ | ✅ | Used in this project (real-time) |
| YOLOv8s | ⚡⚡⚡ | ✅✅ | Good balance |
| SSD MobileNet | ⚡⚡⚡ | ✅ | Lightweight, lower accuracy |
| Faster R-CNN | ⚡ | ✅✅✅ | High accuracy, too slow for real-time |

**YOLOv8n** was chosen for its single-pass architecture, real-time performance (~30–80 FPS), and strong accuracy on the COCO driving classes.

---

## 📂 Project Structure

```
tesla_autopilot_clone/
├── src/
│   ├── detect.py        # Main detection pipeline (image + video)
│   ├── train.py         # Fine-tuning on custom dataset
│   └── evaluate.py      # Precision / Recall / F1 evaluation
├── utils/
│   ├── visualizer.py    # Bounding box drawing + HUD overlay
│   └── metrics.py       # Per-frame statistics tracker
├── notebooks/
│   └── exploration.ipynb  # Interactive Jupyter exploration
├── dataset/
│   └── data.yaml        # Dataset config template
├── output/              # Annotated results saved here
├── assets/              # Sample images for testing
├── requirements.txt
└── README.md
```

---

## 🗂️ Dataset Details

The project uses **COCO 2017** (pre-trained) and can be fine-tuned on any custom driving dataset.

**Detected classes (driving-relevant subset of COCO):**

| Class ID | Label | Colour |
|----------|-------|--------|
| 0 | Person | Yellow |
| 1 | Bicycle | Blue-Orange |
| 2 | Car | Green |
| 3 | Motorcycle | Magenta |
| 5 | Bus | Orange |
| 7 | Truck | Red-Orange |
| 9 | Traffic Light | Cyan |
| 11 | Stop Sign | Red |
| 12 | Parking Meter | Purple |

**Recommended custom datasets:**
- [BDD100K](https://bdd-data.berkeley.edu/) — 100K diverse driving videos
- [KITTI](http://www.cvlibs.net/datasets/kitti/) — Stereo + LiDAR dataset
- [Roboflow Driving Datasets](https://roboflow.com/industries/autonomous-vehicles)

---

## ⚙️ Preprocessing Steps

1. **Resize** all frames to 640×640 (YOLOv8 default input size)
2. **Normalize** pixel values to [0, 1] (done internally by ultralytics)
3. **Augmentation** during training:
   - Random horizontal flip (p=0.5)
   - HSV colour jitter
   - Mosaic augmentation (4-image composite)
   - MixUp (α=0.1)

---

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- pip
- (Optional) CUDA-compatible GPU for faster inference

### Steps

```bash
# 1. Clone / unzip the project
cd tesla_autopilot_clone

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

YOLOv8 weights (`yolov8n.pt`) are **downloaded automatically** on first run.

---

## ▶️ How to Run

### 1. Detect on an image
```bash
python src/detect.py --source assets/sample.jpg --model yolov8n.pt --conf 0.40
```

### 2. Detect on a video file
```bash
python src/detect.py --source path/to/video.mp4 --conf 0.40
```

### 3. Live webcam detection
```bash
python src/detect.py --source 0
```

### 4. Fine-tune on custom data
```bash
python src/train.py --data dataset/data.yaml --model yolov8n.pt --epochs 50 --batch 16
```

### 5. Evaluate on labelled dataset
```bash
python src/evaluate.py --data dataset/images --labels dataset/labels \
                       --model yolov8n.pt --conf 0.40 --iou 0.50
```

All annotated outputs are saved to the `output/` folder.

---

## 📊 Results & Metrics

> *(Results below are representative; run `evaluate.py` on your dataset to get actual numbers.)*

| Metric | Value (YOLOv8n, COCO val) |
|--------|--------------------------|
| Precision | 0.72 |
| Recall | 0.68 |
| F1 Score | 0.70 |
| mAP@0.5 | 0.67 |
| Avg Inference | ~12 ms / frame (GPU) |
| FPS | ~80 FPS (GPU) / ~15 FPS (CPU) |

### Sample Detection Output

Bounding boxes are colour-coded by class with confidence scores displayed. A HUD overlay (top-left) shows live inference time, FPS, and object count.

---

## 🚧 Challenges & Improvements

**Challenges faced:**
- Small object detection (distant pedestrians, far-away traffic lights) reduces recall
- Occlusion and overlapping bounding boxes in dense traffic scenes
- CPU-only inference limits real-time performance

**Potential improvements:**
- Use YOLOv8m or YOLOv8l for higher accuracy at the cost of speed
- Add lane detection using OpenCV Hough transforms
- Integrate depth estimation for 3D bounding boxes
- Add object tracking (SORT/ByteTrack) for consistent IDs across frames
- Deploy with TensorRT for edge/embedded inference (NVIDIA Jetson)

---

## 📄 License

This project is for educational purposes as part of the GUVI AIML course.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [COCO Dataset](https://cocodataset.org/)
- GUVI × HCL AIML Course
