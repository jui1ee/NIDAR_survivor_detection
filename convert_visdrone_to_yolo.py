import os
import shutil
from tqdm import tqdm

# -----------------------------
# CONFIGURE PATHS (UPDATED)
# -----------------------------
visdrone_train_images = "data/raw/train/VisDrone2019-DET-train/images"
visdrone_train_ann    = "data/raw/train/VisDrone2019-DET-train/annotations"

visdrone_val_images   = "data/raw/val/VisDrone2019-DET-val/images"
visdrone_val_ann      = "data/raw/val/VisDrone2019-DET-val/annotations"

yolo_train_images = "data/processed/yolo_dataset/images/train"
yolo_train_labels = "data/processed/yolo_dataset/labels/train"

yolo_val_images   = "data/processed/yolo_dataset/images/val"
yolo_val_labels   = "data/processed/yolo_dataset/labels/val"

os.makedirs(yolo_train_images, exist_ok=True)
os.makedirs(yolo_train_labels, exist_ok=True)
os.makedirs(yolo_val_images, exist_ok=True)
os.makedirs(yolo_val_labels, exist_ok=True)

# -----------------------------
# VISDRONE → YOLO CLASS MAPPING
# -----------------------------
visdrone_to_yolo = {
    1: 0,  # pedestrian → human
    2: 0,  # people → human
}

# -----------------------------
# FUNCTION: Convert single annotation file
# -----------------------------
def convert_visdrone_annotation(txt_path, img_w, img_h):
    yolo_lines = []

    with open(txt_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue

            x, y, w, h = map(float, parts[:4])
            cls = int(parts[5])

            # only keep human classes
            if cls not in visdrone_to_yolo:
                continue

            yolo_cls = visdrone_to_yolo[cls]

            # convert to YOLO normalized format
            x_center = (x + w / 2.0) / img_w
            y_center = (y + h / 2.0) / img_h
            w_yolo   = w / img_w
            h_yolo   = h / img_h

            yolo_lines.append(f"{yolo_cls} {x_center} {y_center} {w_yolo} {h_yolo}")

    return yolo_lines


# -----------------------------
# PROCESS TRAIN/VAL SPLIT
# -----------------------------
def process_split(img_dir, ann_dir, out_img_dir, out_label_dir):
    images = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

    for img_name in tqdm(images, desc=f"Processing {img_dir}"):

        # copy image to YOLO dataset
        shutil.copy(
            os.path.join(img_dir, img_name),
            os.path.join(out_img_dir, img_name)
        )

        # corresponding annotation
        ann_name = img_name.replace(".jpg", ".txt")
        ann_path = os.path.join(ann_dir, ann_name)

        # if no annotation → skip
        if not os.path.exists(ann_path):
            continue

        # image dims
        import cv2
        img = cv2.imread(os.path.join(img_dir, img_name))
        h, w, _ = img.shape

        yolo_lines = convert_visdrone_annotation(ann_path, w, h)

        # save label
        out_label_path = os.path.join(out_label_dir, ann_name)
        with open(out_label_path, "w") as f:
            f.write("\n".join(yolo_lines))


# -----------------------------
# RUN CONVERSION
# -----------------------------
process_split(visdrone_train_images, visdrone_train_ann, yolo_train_images, yolo_train_labels)
process_split(visdrone_val_images, visdrone_val_ann, yolo_val_images, yolo_val_labels)

print("✔️ DONE — YOLO dataset successfully created!")
