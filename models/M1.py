# -*- coding: utf-8 -*-
"""
Model M1: Custom CNN for image classification.
Architecture: Depthwise Separable Conv + SE Attention + BatchNorm + Residual.
Supports both 224x224 and 32x32 input sizes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.SE_Attention import SE


class BlockM1(nn.Module):
    """
    PDP Block with BatchNorm, SE Attention, and Residual connection.
    Flow: PW1(1x1) -> BN -> ReLU -> DW(3x3) -> BN -> ReLU -> PW2(1x1) -> BN -> SE -> + shortcut -> ReLU
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(BlockM1, self).__init__()
        # Pointwise 1: channel mixing
        self.pw1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        # Depthwise 3x3: spatial filtering
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                            padding=1, groups=in_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        # Pointwise 2: project to output channels
        self.pw2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        # SE Attention (reduction_ratio=8 for better capacity on small datasets)
        self.se = SE(out_channels, 8)
        # Shortcut / residual connection
        self.need_proj = (stride != 1) or (in_channels != out_channels)
        if self.need_proj:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.pw1(x)), inplace=True)
        out = F.relu(self.bn2(self.dw(out)), inplace=True)
        out = self.bn3(self.pw2(out))
        out = self.se(out)
        if self.need_proj:
            identity = self.shortcut(identity)
        out = out + identity
        out = F.relu(out, inplace=True)
        return out


class NetM1(nn.Module):
    """
    M1 CNN model for image classification.
    Args:
        n_class: number of output classes (default: 3)
        img_size: input image size, 224 or 32 (default: 224)
    
    Architecture for 224x224:
        Stem(3->32, s=2)  -> 112x112x32
        Block1(32->64, s=2)  -> 56x56x64
        Block2(64->64, s=1)  -> 56x56x64
        Block3(64->128, s=2) -> 28x28x128
        Block4(128->128, s=1)-> 28x28x128
        Block5(128->256, s=2)-> 14x14x256
        Block6(256->256, s=1)-> 14x14x256
        Block7(256->512, s=2)-> 7x7x512
        Head PW(512->1024)   -> 7x7x1024
        GAP -> Dropout -> FC(1024, n_class)
    
    Architecture for 32x32:
        Stem(3->32, s=1)     -> 32x32x32
        Block1(32->64, s=1)  -> 32x32x64
        Block2(64->64, s=1)  -> 32x32x64
        Block3(64->128, s=2) -> 16x16x128
        Block4(128->128, s=1)-> 16x16x128
        Block5(128->256, s=2)-> 8x8x256
        Block6(256->256, s=1)-> 8x8x256
        Block7(256->512, s=2)-> 4x4x512
        Head PW(512->1024)   -> 4x4x1024
        GAP -> Dropout -> FC(1024, n_class)
    """
    def __init__(self, n_class=3, img_size=224):
        super(NetM1, self).__init__()
        self.img_size = img_size

        if img_size == 224:
            # Stem: 224 -> 112 (stride=2)
            self.stem = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
            # Body: 112 -> 56 -> 28 -> 14 -> 7
            self.body = nn.Sequential(
                BlockM1(32, 64, stride=2),     # 112 -> 56
                BlockM1(64, 64, stride=1),     # 56 -> 56
                BlockM1(64, 128, stride=2),    # 56 -> 28
                BlockM1(128, 128, stride=1),   # 28 -> 28
                BlockM1(128, 256, stride=2),   # 28 -> 14
                BlockM1(256, 256, stride=1),   # 14 -> 14
                BlockM1(256, 512, stride=2),   # 14 -> 7
            )
        elif img_size == 32:
            # Stem: 32 -> 32 (stride=1)
            self.stem = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
            # Body: 32 -> 16 -> 8 -> 4
            self.body = nn.Sequential(
                BlockM1(32, 64, stride=1),     # 32 -> 32
                BlockM1(64, 64, stride=1),     # 32 -> 32
                BlockM1(64, 128, stride=2),    # 32 -> 16
                BlockM1(128, 128, stride=1),   # 16 -> 16
                BlockM1(128, 256, stride=2),   # 16 -> 8
                BlockM1(256, 256, stride=1),   # 8 -> 8
                BlockM1(256, 512, stride=2),   # 8 -> 4
            )
        else:
            raise ValueError(f"Unsupported img_size={img_size}. Use 224 or 32.")

        # Head: pointwise conv to expand channels
        self.head = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(1024, n_class)

        # Kaiming initialization for better convergence
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
