from models.fusion import FusedYolo
from ultralytics import YOLO
import torch.nn as nn
import torch
from dataset.llvip import LLVIPDataset


batch_size = 4
epochs = 100

model = FusedYolo(
    YOLO("yolov10x.yaml", task="detect").load("results/train_rgb/weights/best.pt")
)

train_dataset = LLVIPDataset(
    rgb_dir="datasets/visible",
    infra_dir="datasets/infrared",
    label_dir="datasets/visible/labels",
    set="train",
)

val_dataset = LLVIPDataset(
    rgb_dir="datasets/visible",
    infra_dir="datasets/infrared",
    label_dir="datasets/visible/labels",
    set="val",
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=1,
)

# val_loader = torch.utils.data.DataLoader(
#     val_dataset,
#     batch_size=batch_size,
#     shuffle=False,
#     num_workers=1,
# )

for epoch in range(epochs):
    for batch_idx, (x1, x2, targets) in enumerate(train_loader):
        loss = model(x1, x2, targets)
        print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss}")
