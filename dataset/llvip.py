import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

to_tensor = transforms.ToTensor()


class LLVIPDataset(Dataset):
    def __init__(self, rgb_dir, infra_dir, label_dir, transforms=None, set="train"):
        assert set in ["train", "val"]

        self.rgb_dir = f"{rgb_dir}/images/{set}"
        self.infra_dir = f"{infra_dir}/images/{set}"
        self.label_dir = f"{label_dir}/{set}"
        self.transforms = transforms
        self.set = set

        self.rgb_files = os.listdir(self.rgb_dir)
        self.infra_files = os.listdir(self.infra_dir)

        assert len(self.rgb_files) == len(self.infra_files)

        self.img_ids = [os.path.splitext(file)[0] for file in self.rgb_files]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        label_path = f"{self.label_dir}/{self.img_ids[idx]}.txt"
        rgb_path = f"{self.rgb_dir}/{self.img_ids[idx]}.jpg"
        infra_path = f"{self.infra_dir}/{self.img_ids[idx]}.jpg"

        rgb_img = Image.open(rgb_path).convert("RGB")
        infra_img = Image.open(infra_path).convert("L")

        with open(label_path, "r") as f:
            boxes = []
            for line in f.readlines():
                class_id, cx, cy, w, h = map(float, line.split())
                boxes.append([class_id, cx, cy, w, h])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        rgb_img = to_tensor(rgb_img)
        infra_img = to_tensor(infra_img)

        # if self.transforms is not None:
        #     rgb_img, infra_img, boxes = self.transforms(rgb_img, infra_img, boxes)

        return rgb_img, infra_img, boxes
