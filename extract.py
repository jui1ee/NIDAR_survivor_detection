import zipfile, os

paths = [
    "data/raw/train/VisDrone2019-DET-train.zip",
    "data/raw/val/VisDrone2019-DET-val.zip"
]

for p in paths:
    out_dir = os.path.dirname(p)  # extract into the same folder (train/ or val/)
    print("Extracting:", p)
    with zipfile.ZipFile(p, 'r') as zip_ref:
        zip_ref.extractall(out_dir)

print("Done.")
