import torch
import torch.nn as nn

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_dw(inp, oup, stride):
    return nn.Sequential(
        # 深度卷积 (depthwise)
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        
        # 逐点卷积 (pointwise)
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )

class MobileNetV1(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0):
        super(MobileNetV1, self).__init__()
        self.num_classes = num_classes
        
        # 基础通道数
        input_channel = 32
        last_channel = 1024
        
        # 调整通道数
        input_channel = int(input_channel * width_mult)
        last_channel = int(last_channel * width_mult)
        
        # 构建网络
        self.features = nn.Sequential(
            # 第一层是标准卷积
            conv_bn(3, input_channel, 2),
            
            # 深度可分离卷积层
            conv_dw(input_channel, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            
            # 5个相同的层
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            
            # 最后两层
            conv_dw(512, 1024, 2),
            conv_dw(1024, last_channel, 1),
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(last_channel, num_classes),
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

def MobileNet(num_classes=10, width_mult=1.0):
    return MobileNetV1(num_classes, width_mult) 