
```markdown
# NIDAR Survivor Detection

Detect stranded humans in drone footage using YOLOv8.

---

## Project Overview

This repository contains code and utilities for detecting survivors from drone images using the **YOLOv8 object detection framework**.  
The project is part of the NIDAR flood rescue initiative.

- **Input:** Drone images (VisDrone dataset)  
- **Output:** Bounding boxes for humans.  
- **Framework:** YOLOv8 (Ultralytics)  
- **GPU-accelerated training with CUDA**

---

## Repository Structure

```

NIDAR_survivor_detection/
│
├─ check_gpu.py                  # Checks CUDA GPU availability
├─ extract.py                    # Extracts downloaded dataset ZIP files
├─ convert_visdrone_to_yolo.py   # Converts VisDrone labels to YOLO format
├─ data/                         # (ignored by Git, contains raw/processed dataset)
├─ runs/                         # YOLOv8 training results (generated after training)
└─ README.md



---

## Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/jui1ee/NIDAR_survivor_detection.git
cd NIDAR_survivor_detection
````

2. **Create conda environment**

```bash
conda create -n dronecv python=3.10 -y
conda activate dronecv
```

3. **Install dependencies**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics opencv-python numpy matplotlib tqdm pandas seaborn jupyter
```

4. **Check GPU**

```bash
python check_gpu.py
```

---

## Dataset Preparation

1. **Download VisDrone dataset** and place the ZIPs under `data/raw/`.
2. **Extract dataset**:

```bash
python extract.py
```

3. **Convert labels to YOLO format**:

```bash
python convert_visdrone_to_yolo.py
```

4. **Organize dataset structure**:

```
data/processed/yolo_dataset/
├─ images/
│  ├─ train/
│  └─ val/
└─ labels/
   ├─ train/
   └─ val/
```

5. **Create dataset config** (`dataset.yaml`):

```yaml
path: data/processed/yolo_dataset
train: images/train
val: images/val
names:
  0: human
```

---

## Training YOLOv8

Example command for a **quick test run** (format it according to Mac/Linux/WSL):

```bash
yolo detect train \
    model=yolov8n.pt \
    data=data/processed/yolo_dataset/dataset.yaml \
    epochs=5 \
    imgsz=512 \
    batch=4 \
    device=0 \
    half=True \
    workers=2 \
    project=runs/train \
    name=survivor_detection_test \
    exist_ok=True
```

* Check `runs/train/survivor_detection_test/` for results, plots, and weights (`best.pt` for inference).

---

## Inference / Prediction

After training, you can detect survivors on new images:

```bash
yolo detect predict \
    model=runs/train/survivor_detection_test/weights/best.pt \
    source=data/raw/val/VisDrone2019-DET-val/images
```

Results with bounding boxes will be saved in `runs/detect/predict/`.

---

## Metrics

* **mAP50** → Primary detection accuracy metric
* **Precision / Recall** → Box-level evaluation
* Loss curves are plotted in `results.png` in the training run folder

---

## Notes

* The repository does **not store raw or processed datasets** due to size.
* You can adjust **epochs, batch size, image size** for faster experimentation or better accuracy.
* Ensure your GPU has sufficient memory (≥4GB recommended).

```

```
