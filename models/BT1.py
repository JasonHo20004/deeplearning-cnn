# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 08:29:38 2026

@author: tuann
"""
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
class moduleDP(nn.Module):
    def __init__(self,in_channels,out_channels, kernel_size=3,stride=1,padding=0):
        super(moduleDP, self).__init__()
        #depthwise
        self.dw = nn.Conv2d(in_channels = in_channels, out_channels=in_channels, kernel_size=kernel_size,stride=stride, padding=padding,groups=in_channels)
        self.pw = nn.Conv2d(in_channels = in_channels, out_channels=out_channels, kernel_size=1)
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return F.relu(x)

class NetBT2(nn.Module):
    def __init__(self, n_class=10):
        super(NetBT2, self).__init__()
        self.DP1 = moduleDP(in_channels = 9, out_channels=256,stride=1,padding=1)
        self.DP2 = moduleDP(in_channels = 256, out_channels=128, kernel_size=5,stride=2, padding=2)
        self.DP3 = moduleDP(in_channels = 128, out_channels=256, kernel_size=3,stride=1, padding=1)
        self.DP4 = moduleDP(in_channels = 256, out_channels=512, kernel_size=5,stride=1, padding=2)
        self.DP5 = moduleDP(in_channels = 512, out_channels=256, kernel_size=3,stride=2, padding=1)
        self.DP6 = moduleDP(in_channels = 256, out_channels=1024, kernel_size=3,stride=1, padding=1)
        self.DP7 = moduleDP(in_channels = 1024, out_channels=512, kernel_size=5,stride=2, padding=2)
        self.DP8 = moduleDP(in_channels = 512, out_channels=512, kernel_size=1,stride=2, padding=0)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(512, n_class)#CIFAR-10
    def forward(self, x):
        x_gray=T.functional.rgb_to_grayscale(x)
        Gau1 = T.GaussianBlur(kernel_size=(3,3), sigma=(0.5, 0.5))
        Gau2 = T.GaussianBlur(kernel_size=(3,3), sigma=(0.7, 0.7))
        Gau3 = T.GaussianBlur(kernel_size=(3,3), sigma=(0.9, 0.9))
        Gau4 = T.GaussianBlur(kernel_size=(3,3), sigma=(1.1, 1.1))
        Gau5 = T.GaussianBlur(kernel_size=(3,3), sigma=(1.3, 1.3))
        Gau6 = T.GaussianBlur(kernel_size=(3,3), sigma=(1.5, 1.5))
        x_gau1 = Gau1(x_gray)
        x_gau2 = Gau2(x_gray)
        x_gau3 = Gau3(x_gray)
        x_gau4 = Gau4(x_gray)
        x_gau5 = Gau5(x_gray)
        x_gau6 = Gau6(x_gray)
        x = torch.cat((x,x_gau1,x_gau2,x_gau3,x_gau4,x_gau5,x_gau6),dim=1)
        x = self.DP1(x)
        x = self.DP2(x)
        x = self.DP3(x)
        x = self.DP4(x)
        x = self.DP5(x)
        x = self.DP6(x)
        x = self.DP7(x)
        x = self.DP8(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #print(x.size())
        x = self.fc(x)
        return x
class NetBT1(nn.Module):
    def __init__(self):
        super(NetBT1, self).__init__()
        # self.Gau1 = T.GaussianBlur(kernel_size=(3,3), sigma=(0.5, 0.5))
        # self.Gau2 = T.GaussianBlur(kernel_size=(3,3), sigma=(0.7, 0.7))
        # self.Gau3 = T.GaussianBlur(kernel_size=(3,3), sigma=(0.9, 0.9))
        # self.Gau4 = T.GaussianBlur(kernel_size=(3,3), sigma=(1.1, 1.1))
        # self.Gau5 = T.GaussianBlur(kernel_size=(3,3), sigma=(1.3, 1.3))
        # self.Gau6 = T.GaussianBlur(kernel_size=(3,3), sigma=(1.5, 1.5))
        self.conv1 = nn.Conv2d(in_channels = 9, out_channels=256, kernel_size=3,stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels = 256, out_channels=128, kernel_size=5,stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels=256, kernel_size=3,stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels = 256, out_channels=512, kernel_size=5,stride=1, padding=2)
        self.conv5 = nn.Conv2d(in_channels = 512, out_channels=256, kernel_size=3,stride=2, padding=1)
        self.conv6 = nn.Conv2d(in_channels = 256, out_channels=1024, kernel_size=3,stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels = 1024, out_channels=512, kernel_size=5,stride=2, padding=2)
        self.conv8 = nn.Conv2d(in_channels = 512, out_channels=512, kernel_size=1,stride=2, padding=0)
        
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(512, 10)#CIFAR-10
    def forward(self, x):
        x_gray=T.functional.rgb_to_grayscale(x)
        Gau1 = T.GaussianBlur(kernel_size=(3,3), sigma=(0.5, 0.5))
        Gau2 = T.GaussianBlur(kernel_size=(3,3), sigma=(0.7, 0.7))
        Gau3 = T.GaussianBlur(kernel_size=(3,3), sigma=(0.9, 0.9))
        Gau4 = T.GaussianBlur(kernel_size=(3,3), sigma=(1.1, 1.1))
        Gau5 = T.GaussianBlur(kernel_size=(3,3), sigma=(1.3, 1.3))
        Gau6 = T.GaussianBlur(kernel_size=(3,3), sigma=(1.5, 1.5))
        x_gau1 = Gau1(x_gray)
        x_gau2 = Gau2(x_gray)
        x_gau3 = Gau3(x_gray)
        x_gau4 = Gau4(x_gray)
        x_gau5 = Gau5(x_gray)
        x_gau6 = Gau6(x_gray)
        x = torch.cat((x,x_gau1,x_gau2,x_gau3,x_gau4,x_gau5,x_gau6),dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #print(x.size())
        x = self.fc(x)
        return x
        
        