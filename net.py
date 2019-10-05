#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 12:38:20 2018

@author: dengbin
"""

import torch
import torch.nn as nn


############################ network design
class CNNEncoder(nn.Module):
    """Deep Embedding Module"""
    def __init__(self,input_channels,feature_dim=64):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(input_channels,feature_dim,kernel_size=1,padding=0),
                        nn.BatchNorm2d(feature_dim, momentum=1, affine=True),
                        nn.ReLU())
        self.layer2 = nn.Sequential(
                        nn.Conv2d(feature_dim,feature_dim,kernel_size=1,padding=0),
                        nn.BatchNorm2d(feature_dim, momentum=1, affine=True),
                        nn.ReLU())
        self.layer3 = nn.Sequential(
                        nn.Conv2d(feature_dim,feature_dim,kernel_size=1,padding=0),
                        nn.BatchNorm2d(feature_dim, momentum=1, affine=True),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv2d(feature_dim,feature_dim,kernel_size=1,padding=0),
                        nn.BatchNorm2d(feature_dim, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out # 64

class RelationNetwork(nn.Module):
    """Deep Metric Module"""
    def __init__(self, patch_size,feature_dim=64):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(feature_dim*2,feature_dim,kernel_size=1,padding=0),
                        nn.BatchNorm2d(feature_dim, momentum=1, affine=True),
                        nn.ReLU())
        self.layer2 = nn.Sequential(
                        nn.Conv2d(feature_dim,feature_dim,kernel_size=1,padding=0),
                        nn.BatchNorm2d(feature_dim, momentum=1, affine=True),
                        nn.ReLU())
        self.layer3 = nn.Conv2d(feature_dim,1,kernel_size=patch_size,padding=0)
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0),-1)
        out = torch.sigmoid(out)
        return out

######################################################## Initiate weights of net
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, 0.05)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())
    else:
        pass