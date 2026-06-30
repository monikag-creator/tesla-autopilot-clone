# Tesla Autopilot Clone — YOLOv8 Object Detection

A real-time lane and object detection system inspired by Tesla Autopilot, built using YOLOv8 to detect vehicles, pedestrians, and traffic signs from dashcam footage.

## Problem Statement
Road accidents caused by human error and delayed reaction time remain one of the leading causes of fatalities worldwide. Most existing driver-assist systems rely on expensive proprietary hardware, making advanced safety features inaccessible for affordable vehicles. There is a need for an open, lightweight computer vision system that can detect road hazards — vehicles, pedestrians, lanes, and traffic signs — in real time using just a camera feed.

## Tech Stack
- **Language:** Python
- **Object Detection:** YOLOv8 (Ultralytics)
- **Computer Vision:** OpenCV
- **Data Handling:** NumPy, Pandas
- **Visualization:** Matplotlib
- **Environment:** Google Colab / Jupyter Notebook

## Approach
1. Collected and curated a dataset of dashcam footage containing vehicles, pedestrians, and traffic signs.
2. Preprocessed and annotated the dataset in YOLO format for object detection training.
3. Fine-tuned a pre-trained YOLOv8 model on the custom dataset using transfer learning.
4. Implemented real-time inference on video streams with bounding boxes, class labels, and confidence scores overlaid on each frame.
5. Evaluated model performance using mean Average Precision (mAP) on a held-out test set.
6. Achieved an **mAP score of 0.82**, indicating strong detection accuracy across object classes.

## Business Solution
This system demonstrates how affordable, camera-only computer vision can power core ADAS (Advanced Driver Assistance System) features — lane detection, collision warnings, and pedestrian alerts — without expensive LiDAR or radar hardware. It can be adapted by automotive startups, fleet management companies, or insurance providers to build low-cost driver-safety add-ons, reduce accident rates, and lower insurance risk through real-time hazard alerts.

## Results
| Metric | Score |
|---|---|
| mAP (mean Average Precision) | 0.82 |

## How to Run
```bash
git clone https://github.com/monikag-creator/tesla-autopilot-clone.git
cd tesla-autopilot-clone
pip install -r requirements.txt
python detect.py --source <video_path>
```

## Author
**Monika G** — [LinkedIn](https://www.linkedin.com/in/monika-g-4a2904388) | [GitHub](https://github.com/monikag-creator)
