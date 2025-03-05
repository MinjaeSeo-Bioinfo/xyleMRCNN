import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.ops import misc

# SE 블록 구현
class SEBlock(nn.Module):
    def __init__(self, in_channels, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1,1))
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // r),
            nn.ReLU(),
            nn.Linear(in_channels // r, in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        x = self.squeeze(x)
        x = x.view(b, c)
        x = self.excitation(x)
        x = x.view(b, c, 1, 1)
        return x

# 기본 컨볼루션 블록
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6()
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x

# Depthwise Separable Convolution
class Depthwise(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(),
        )
        self.seblock = SEBlock(out_channels)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        se_weight = self.seblock(x)
        x = x * se_weight
        return x

# SE-ResNet 백본
class SEResBackbone(nn.Module):
    def __init__(self, backbone_name, pretrained):
        super().__init__()
        body = models.resnet.__dict__[backbone_name](
            pretrained=pretrained, norm_layer=misc.FrozenBatchNorm2d)
        
        for name, parameter in body.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
                
        self.body = nn.ModuleDict(d for i, d in enumerate(body.named_children()) if i < 8)
        in_channels = 2048
        self.out_channels = 256
        
        # SE 블록 추가
        self.se_block = SEBlock(in_channels)
        
        # 내부 블록 모듈 및 레이어 블록 모듈
        self.inner_block_module = BasicConv2d(in_channels, self.out_channels, kernel_size=1, padding=0)
        self.layer_block_module = Depthwise(self.out_channels, self.out_channels)
        
        # 가중치 초기화
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 백본을 통한 특징 추출
        for module in self.body.values():
            x = module(x)
        
        # SE 블록 적용
        se_weight = self.se_block(x)
        x = x * se_weight
        
        # 추가 처리
        x = self.inner_block_module(x)
        x = self.layer_block_module(x)
        return x

