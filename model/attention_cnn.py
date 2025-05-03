import torch
import torch.nn as nn
import torch.utils
from torchvision.transforms import transforms


class SelfAttn(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttn, self).__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.size()

        Q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # (B, H*W, C//8)
        K = self.key(x).view(B, -1, H * W)  # (B, C//8, H*W)
        V = self.value(x).view(B, -1, H * W)  # (B, C, H*W)

        attention = self.softmax(torch.bmm(Q, K))  # (B, H*W, H*W)

        out = torch.bmm(V, attention.permute(0, 2, 1))  # (B, C, H*W)
        out = out.view(B, C, H, W)

        return out + x


class AttentionCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(AttentionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.attn1 = SelfAttn(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.attn2 = SelfAttn(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.attn3 = SelfAttn(256)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        self.activate = torch.relu

    def forward(self, x):
        x = self.activate(self.conv1(x))
        x = self.attn1(x)
        x = self.activate(self.conv2(x))
        x = self.attn2(x)
        x = self.activate(self.conv3(x))
        x = self.attn3(x)
        x = self.pool(x).view(x.shape[0], -1)
        return self.fc(x)
