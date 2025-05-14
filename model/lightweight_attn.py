import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """轻量级通道注意力机制"""
    def __init__(self, in_channels, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 使用更少的参数实现通道注意力
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return torch.sigmoid(out)

class SpatialAttention(nn.Module):
    """轻量级空间注意力机制"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        
    def forward(self, x):
        # 沿通道维度计算平均值和最大值，并拼接
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return torch.sigmoid(out)

class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride, 
            padding=1, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, 
            padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class LightweightBlock(nn.Module):
    """轻量级模块，结合深度可分离卷积、注意力机制和跳跃连接"""
    def __init__(self, in_channels, out_channels, stride=1, use_attention=True):
        super(LightweightBlock, self).__init__()
        self.use_attention = use_attention
        self.conv = DepthwiseSeparableConv(in_channels, out_channels, stride)
        
        # 注意力模块
        if use_attention:
            self.ca = ChannelAttention(out_channels)
            self.sa = SpatialAttention()
            
        # 跳跃连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = self.conv(x)
        
        if self.use_attention:
            # 应用通道注意力
            ca_weight = self.ca(out)
            out = out * ca_weight
            
            # 应用空间注意力
            sa_weight = self.sa(out)
            out = out * sa_weight
        
        # 添加跳跃连接
        out = out + self.shortcut(x)
        return out

class LightweightAttnNet(nn.Module):
    """轻量级注意力网络"""
    def __init__(self, num_classes=10, width_mult=0.5):
        super(LightweightAttnNet, self).__init__()
        
        # 基础通道数（较小以减少参数）
        base_channels = int(32 * width_mult)
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 轻量级模块堆叠
        self.layer1 = self._make_layer(base_channels, base_channels*2, stride=1)
        self.layer2 = self._make_layer(base_channels*2, base_channels*4, stride=2)
        self.layer3 = self._make_layer(base_channels*4, base_channels*8, stride=2)
        
        # 全局池化和分类器
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(base_channels*8, num_classes)
        
        # 权重初始化
        self._initialize_weights()
        
    def _make_layer(self, in_channels, out_channels, stride):
        return LightweightBlock(in_channels, out_channels, stride, use_attention=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

def LightweightAttn(num_classes=10, width_mult=0.5):
    """创建轻量级注意力网络实例"""
    return LightweightAttnNet(num_classes, width_mult) 