import torch.nn as nn
import torch


class FusedYolo(nn.Module):
    def __init__(self, yolo_model):
        super().__init__()
        self.conv_rgb = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=80,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(80),
            nn.ReLU(),
        )
        self.conv_infra = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=80,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(80),
            nn.ReLU(),
        )
        self.fuse = nn.Conv2d(160, 3, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model = yolo_model

    def forward(self, x1, x2):
        x1_features = self.conv_rgb(x1)
        x2_features = self.conv_infra(x2)

        x = torch.cat((x1_features, x2_features), dim=1)
        x = self.fuse(x)
        return self.model(x)
