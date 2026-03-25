import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.SE_Attention import *
class BlockBT3(nn.Module):
    def __init__(self, in_channels, out_channels, s):
        super(BlockBT3, self).__init__()
        
        self.Pw1 = nn.Conv2d(in_channels = in_channels, out_channels=in_channels, kernel_size=1)
        self.Dw = nn.Conv2d(in_channels = in_channels, out_channels=in_channels, kernel_size=3,stride=s,padding=1,groups=in_channels)
        self.Pw2 = nn.Conv2d(in_channels = in_channels, out_channels=out_channels, kernel_size=1)
        self.PwR = nn.Conv2d(in_channels = in_channels, out_channels=out_channels, kernel_size=1,stride=s)
        self.SE = SE(out_channels,16)
        self.s = s
    def forward(self, x):
        Pw1 = F.relu(self.Pw1(x))
        Dw = F.relu(self.Dw(Pw1))
        Pw2 = F.relu(self.Pw2(Dw))
        att = self.SE(Pw2)
        PDP = Pw2 * att
        #print(PDP.size())
        #print(x.size())
        #print(PDP.size()==x.size())
        if self.s == 1 and PDP.size()==x.size():
            x = PDP + x #residual (ResNet)
        else:
            PwR = self.PwR(x)
            #print(PwR.size())
            #print(PDP.size())
            x = PDP + PwR
        return x
        
class NetBT3(nn.Module):
    def __init__(self, n_class=10):
        super(NetBT3, self).__init__()
        # Stem: 224x224x3 → 112x112x32
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        # Body 1: 112x112x32 → 112x112x64,  s=1
        self.body1 = BlockBT3(in_channels=32,  out_channels=64,  s=1)
        # Body 2: 112x112x64 → 56x56x64,    s=2
        self.body2 = BlockBT3(in_channels=64,  out_channels=64,  s=2)
        # Body 3: 56x56x64 → 56x56x128,     s=1
        self.body3 = BlockBT3(in_channels=64,  out_channels=128, s=1)
        # Body 4: 56x56x128 → 56x56x128,    s=1
        self.body4 = BlockBT3(in_channels=128, out_channels=128, s=1)
        # Body 5: 56x56x128 → 28x28x256,    s=2
        self.body5 = BlockBT3(in_channels=128, out_channels=256, s=2)
        # Body 6: 28x28x256 → 28x28x256,    s=1
        self.body6 = BlockBT3(in_channels=256, out_channels=256, s=1)
        # Body 7: 28x28x256 → 28x28x256,    s=1
        self.body7 = BlockBT3(in_channels=256, out_channels=256, s=1)
        # Body 8: 28x28x256 → 14x14x512,    s=2
        self.body8 = BlockBT3(in_channels=256, out_channels=512, s=2)
        # Body 9: 14x14x512 → 7x7x512,     s=2
        self.body9 = BlockBT3(in_channels=512, out_channels=512, s=2)
        # Head: Pointwise Conv 7x7x512 → 7x7x1024
        self.head_pw = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        # GAP → FC
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(1024, n_class)

    def forward(self, x):
        x = self.stem(x)
        x = self.body1(x)
        x = self.body2(x)
        x = self.body3(x)
        x = self.body4(x)
        x = self.body5(x)
        x = self.body6(x)
        x = self.body7(x)
        x = self.body8(x)
        x = self.body9(x)
        x = self.head_pw(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x