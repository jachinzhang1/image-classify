import torch
import torch.nn as nn
import math

# Swish激活函数
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# 挤压激发(SE)模块
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# MBConv块 (Mobile Inverted Residual Bottleneck)
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25, drop_rate=0.2):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.drop_rate = drop_rate
        self.use_residual = in_channels == out_channels and stride == 1
        
        # 点态卷积扩展通道
        exp_channels = in_channels * expand_ratio
        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_channels, exp_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(exp_channels),
            Swish()
        ) if expand_ratio != 1 else nn.Identity()
        
        # 深度可分离卷积
        self.depth_conv = nn.Sequential(
            nn.Conv2d(exp_channels, exp_channels, kernel_size=kernel_size, stride=stride, 
                      padding=kernel_size//2, groups=exp_channels, bias=False),
            nn.BatchNorm2d(exp_channels),
            Swish()
        )
        
        # 挤压激发模块
        self.se = SELayer(exp_channels, reduction=int(1/se_ratio)) if se_ratio > 0 else nn.Identity()
        
        # 点态卷积投影回原始通道
        self.project_conv = nn.Sequential(
            nn.Conv2d(exp_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Dropout层用于随机深度
        self.dropout = nn.Dropout2d(drop_rate) if drop_rate > 0 else nn.Identity()
    
    def forward(self, x):
        identity = x
        
        # 扩展
        x = self.expand_conv(x)
        
        # 深度卷积
        x = self.depth_conv(x)
        
        # 挤压激发
        x = self.se(x)
        
        # 投影
        x = self.project_conv(x)
        
        # 残差连接
        if self.use_residual:
            if self.training and self.drop_rate > 0:
                if torch.rand(1) < self.drop_rate:
                    return identity
            x = identity + x
            
        return x

# EfficientNet-B0 模型
class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0, depth_mult=1.0, dropout_rate=0.2):
        super(EfficientNetB0, self).__init__()
        
        # B0 的基础配置
        settings = [
            # t, c, n, s, k
            # expand_ratio, output_channels, num_repeats, stride, kernel_size
            [1, 16, 1, 1, 3],   # MBConv1_3x3, SE
            [6, 24, 2, 2, 3],   # MBConv6_3x3, SE
            [6, 40, 2, 2, 5],   # MBConv6_5x5, SE
            [6, 80, 3, 2, 3],   # MBConv6_3x3, SE
            [6, 112, 3, 1, 5],  # MBConv6_5x5, SE
            [6, 192, 4, 2, 5],  # MBConv6_5x5, SE
            [6, 320, 1, 1, 3]   # MBConv6_3x3, SE
        ]
        
        # 放大网络宽度
        out_channels = self._round_filters(32, width_mult)
        features = [nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            Swish()
        )]
        
        in_channels = out_channels
        
        # 构建MBConv块
        for t, c, n, s, k in settings:
            out_channels = self._round_filters(c, width_mult)
            repeats = self._round_repeats(n, depth_mult)
            
            for i in range(repeats):
                stride = s if i == 0 else 1
                features.append(MBConvBlock(in_channels, out_channels, kernel_size=k, stride=stride, expand_ratio=t))
                in_channels = out_channels
        
        # 最后一个卷积层
        last_channels = self._round_filters(1280, width_mult)
        features.append(nn.Sequential(
            nn.Conv2d(in_channels, last_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(last_channels),
            Swish()
        ))
        
        self.features = nn.Sequential(*features)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(last_channels, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def _round_filters(self, filters, width_mult):
        """计算缩放后的滤波器数量"""
        multiplier = width_mult
        divisor = 8
        filters *= multiplier
        min_depth = divisor
        new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
        if new_filters < 0.9 * filters:  # 防止舍入造成较大差异
            new_filters += divisor
        return int(new_filters)
    
    def _round_repeats(self, repeats, depth_mult):
        """计算缩放后的重复次数"""
        return int(math.ceil(depth_mult * repeats))
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def EfficientNet(num_classes=10, variant='b0'):
    """
    创建EfficientNet模型
    目前仅支持B0变体
    """
    if variant.lower() == 'b0':
        return EfficientNetB0(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported EfficientNet variant: {variant}") 