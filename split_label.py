import os
import shutil


train_label_path = "dataset/infrared/labels/train"
val_label_path = "dataset/infrared/labels/val"
os.makedirs(train_label_path, exist_ok=True)
os.makedirs(val_label_path, exist_ok=True)

train_ids = [
    filename.split(".")[0] for filename in os.listdir("dataset/infrared/images/train")
]
val_ids = [
    filename.split(".")[0] for filename in os.listdir("dataset/infrared/images/val")
]

for id in train_ids:
    shutil.move(
        os.path.join("dataset/infrared/labels", f"{id}.txt"),
        os.path.join(train_label_path, f"{id}.txt"),
    )

for id in val_ids:
    shutil.move(
        os.path.join("dataset/infrared/labels", f"{id}.txt"),
        os.path.join(val_label_path, f"{id}.txt"),
    )
