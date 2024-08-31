from ultralytics import YOLO
import torch.nn as nn
import numpy as np
import cv2
from torchsummary import summary


def hook_fn(module, input, output):
    print(f"{input[0].shape} : {output[0].shape}")


# Load a pre-trained YOLOv10n model
model = YOLO("yolov10x.pt")
new_input_layer = nn.Conv2d(
    in_channels=3,
    out_channels=80,
    kernel_size=(3, 3),
    stride=(1, 1),
    padding=(1, 1),
    bias=False,
)
summary(
    model,
    input_size=(3, 1024, 1024),
)
# new_input_layer = model.model.model[0].conv
# model.model.model[0].conv = new_input_layer

# img = cv2.imread("./dataset/LLVIP/visible/train/010002.jpg")  # [:, :, -1]
# # new_channel = np.zeros((img.shape[0], img.shape[1], 1))
# # img = np.concatenate((img, new_channel), axis=-1)
# # img = np.expand_dims(img, axis=-1)

# for layer in model.children():
#     #     print(layer)
#     layer.register_forward_hook(hook_fn)

# print(model.model)
# # print(img.shape)
# # Perform object detection on an image
# results = model(img)

# # Display the results
# # results[0].show()
