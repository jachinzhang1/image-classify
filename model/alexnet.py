import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # Modified from 11x11
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # Modified from 3x3
            nn.Conv2d(64, 192, kernel_size=3, padding=1),  # Modified from 5x5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # Modified from 3x3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # Modified from 3x3
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Changed from (6,6)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256, 4096),  # Changed from 256*6*6
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
