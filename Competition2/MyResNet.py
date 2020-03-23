import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.nn as nn
from torch.nn import *
from torch.nn import Sequential

def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1)

def conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride)

class BasicBlock(nn.Module):
    
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        
        super(BasicBlock, self).__init__()
        
        self.convolution_layer1 = conv3x3(in_channel, out_channel, stride = stride)
        self.norm_layer1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        
        self.convolution_layer2 = conv3x3(out_channel, out_channel)
        self.norm_layer2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.norm_layer1(self.convolution_layer1(x))
        out = self.relu(out)
        out = self.norm_layer2(self.convolution_layer2(out))

        if self.downsample is not None:
            x_copy = self.downsample(x)
        else:
            x_copy = x

        out += x_copy
        out = self.relu(out)

        return out

class MyResNet(nn.Module):

    def __init__(self, block, layers, num_classes=2300, outs = [64, 128, 256, 512]):
        super(MyResNet, self).__init__()

        self.in_channel = 64
        self.convolution_layer = nn.Conv2d(3, self.in_channel, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.norm_layer = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.block_layers = Sequential(self.make_layer(block, outs[0], layers[0]),
                                        self.make_layer(block, outs[1], layers[1], stride=1),
                                        self.make_layer(block, outs[2], layers[2], stride=1),
                                        self.make_layer(block, outs[3], layers[3], stride=2))
        
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.out = Sequential(Linear(512, 4096),
                              BatchNorm1d(4096),
                              ReLU(),
                              Linear(4096, 4096),
                              BatchNorm1d(4096),
                              ReLU(),
                              Linear(4096, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, block, channel, blocks, stride=1):
        
        if stride != 1 or self.in_channel != channel:
            downsample = nn.Sequential(
                conv1x1(self.in_channel, channel, stride = stride),
                nn.BatchNorm2d(channel),
            )
        else:
            downsample = None

        layers = []
        layers.append(block(self.in_channel, channel, stride, downsample))
        
        self.in_channel = channel
        
        for i in range(1, blocks):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.convolution_layer(x)
        x = self.relu(self.norm_layer(x))

        x = self.maxpool(x)

        x = self.adaptive_avg_pool(self.block_layers(x))

        x = torch.flatten(x, 1)
        
        x = self.out(x)

        return x
    
    def embedding(self, x):

        x = self.convolution_layer(x)
        x = self.relu(self.norm_layer(x))

        x = self.maxpool(x)

        x = self.adaptive_avg_pool(self.block_layers(x))
        
        x = torch.flatten(x, 1)
        
        return x