import torch
import torchvision
from torch import nn
import torch.nn.functional as F

import os, sys
from fpn import FPN

import numpy as np

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

class FPN_dual(nn.Module):
    def __init__(self):
        super(FPN_dual, self).__init__()
        net = FPN()
        net.load_state_dict(torch.load('ckpt/pretrained_duts.pth'))
        net2 = FPN()

        ##### source
        self.source_layer0 = net.layer0
        self.source_layer1 = net.layer1
        self.source_layer2 = net.layer2
        self.source_layer3 = net.layer3
        self.source_layer4 = net.layer4

        ##### target
        self.target_layer0 = net2.layer0
        self.target_layer1 = net2.layer1
        self.target_layer2 = net2.layer2
        self.target_layer3 = net2.layer3
        self.target_layer4 = net2.layer4

        self.target_convert5 = net2.convert5
        self.target_convert4 = net2.convert4
        self.target_convert3 = net2.convert3
        self.target_convert2 = net2.convert2
        self.target_convert1 = net2.convert1

        self.target_feature_upscore5 = net2.feature_upscore5
        self.target_feature_upscore4 = net2.feature_upscore4
        self.target_feature_upscore3 = net2.feature_upscore3
        self.target_feature_upscore2 = net2.feature_upscore2

        self.target_predict5 = BoundaryModule(5)
        self.target_predict4 = BoundaryModule(4)
        self.target_predict3 = BoundaryModule(3)
        self.target_predict2 = BoundaryModule(2)
        self.target_predict1 = BoundaryModule(1)


        self.target_layer_attention_layer1 = Attention(64, 64, 128)
        self.target_layer_attention_layer2 = Attention(256, 64, 128)
        self.target_layer_attention_layer3 = Attention(512, 32, 64)
        self.target_layer_attention_layer4 = Attention(1024, 16, 32)
        self.target_layer_attention_layer5 = Attention(2048, 16, 32)

        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout) or isinstance(m, nn.PReLU):
                m.inplace = True
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        
        ##### source
        source_layer1 = self.source_layer0(x)
        source_layer2 = self.source_layer1(source_layer1)
        source_layer3 = self.source_layer2(source_layer2)
        source_layer4 = self.source_layer3(source_layer3)
        source_layer5 = self.source_layer4(source_layer4)

        x_size = x.size()[2:]
        l2_size = source_layer2.size()[2:]
        l3_size = source_layer3.size()[2:]
        l4_size = source_layer4.size()[2:]

        ##### target
        target_layer1 = self.target_layer0(x)
        target_layer1 = target_layer1 + self.target_layer_attention_layer1(source_layer1)

        target_layer2 = self.target_layer1(target_layer1)
        target_layer2 = target_layer2 + self.target_layer_attention_layer2(source_layer2)

        target_layer3 = self.target_layer2(target_layer2)
        target_layer3 = target_layer3 + self.target_layer_attention_layer3(source_layer3)

        target_layer4 = self.target_layer3(target_layer3)
        target_layer4 = target_layer4 + self.target_layer_attention_layer4(source_layer4)

        target_layer5 = self.target_layer4(target_layer4)
        target_layer5 = target_layer5 + self.target_layer_attention_layer5(source_layer5)


        x_size = x.size()[2:]
        l2_size = target_layer2.size()[2:]
        l3_size = target_layer3.size()[2:]
        l4_size = target_layer4.size()[2:]

        target_feature5 = self.target_convert5(target_layer5)

        target_feature4 = self.target_feature_upscore5(target_feature5)
        #target_feature4 = target_feature4[:, :, 1: 1 + l4_size[0], 1:1 + l4_size[1]]
        target_feature4 = self.target_convert4(target_layer4) + target_feature4

        target_feature3 = self.target_feature_upscore4(target_feature4)
        target_feature3 = target_feature3[:, :, 1: 1 + l3_size[0], 1:1 + l3_size[1]]
        target_feature3 = self.target_convert3(target_layer3) + target_feature3

        target_feature2 = self.target_feature_upscore3(target_feature3)
        target_feature2 = target_feature2[:, :, 1: 1 + l2_size[0], 1:1 + l2_size[1]]
        target_feature2 = self.target_convert2(target_layer2) + target_feature2

        target_feature1 = self.target_feature_upscore2(target_feature2)
        target_feature1 = self.target_convert1(target_layer1) + target_feature1

        target_predict5_boundary, target_predict5_interior, target_predict5 = self.target_predict5(target_feature5)
        target_predict4_boundary, target_predict4_interior, target_predict4 = self.target_predict4(target_feature4)
        target_predict3_boundary, target_predict3_interior, target_predict3 = self.target_predict3(target_feature3)
        target_predict2_boundary, target_predict2_interior, target_predict2 = self.target_predict2(target_feature2)
        target_predict1_boundary, target_predict1_interior, target_predict1 = self.target_predict1(target_feature1)


        if self.training:
            return target_predict1, target_predict2, target_predict3, target_predict4, target_predict5, \
            target_predict1_boundary, target_predict2_boundary, target_predict3_boundary, target_predict4_boundary, target_predict5_boundary,\
            target_predict1_interior, target_predict2_interior, target_predict3_interior, target_predict4_interior, target_predict5_interior
        return F.sigmoid(target_predict1)#, F.sigmoid(target_predict1_boundary), F.sigmoid(target_predict1_interior)

class Attention(nn.Module):
    def __init__(self, channel, height, width):
        super(Attention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.AvgPool2d((height, width)),
            nn.Softmax(dim=1)
        )

        self.spatial_attention = nn.Softmax(dim=2)

        self.conv = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, channel, kernel_size=1, padding=0), nn.BatchNorm2d(channel), nn.PReLU(),
        )

    def forward(self, x):
        ca = self.channel_attention(x)

        temp = x.view([x.size(0), x.size(1), -1])
        sa = self.spatial_attention(temp)
        sa = sa.view([x.size(0), x.size(1), x.size(2), x.size(3)])

        x_attention = sa * ca

        return self.conv(x_attention) + x_attention


class BoundaryModule(nn.Module):
    def __init__(self, stage):
        super(BoundaryModule, self).__init__()

        if stage == 5 or stage == 4:
            stride_sp = 16
            kernel_sp = 32
        elif stage == 3:
            stride_sp = 8
            kernel_sp = 16
        else:
            stride_sp = 4
            kernel_sp = 8

        self.stride_sp = stride_sp

        self.boundary = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.ConvTranspose2d(1, 1, kernel_sp, stride=stride_sp, bias=False),
        )

        self.interior = nn.Sequential(
            ISD(),
            nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.ConvTranspose2d(1, 1, kernel_sp, stride=stride_sp, bias=False),
        )

        self.compensation = nn.Sequential(
            ISD(),
            nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.BatchNorm2d(128), nn.PReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.ConvTranspose2d(1, 1, kernel_sp, stride=stride_sp, bias=False),
        )

    def forward(self, x):

        index = int(self.stride_sp/2)

        boundary = self.boundary(x)
        boundary = boundary[:,:,index: index+256, index: index+512]
        interior = self.interior(x)
        interior = interior[:, :, index: index + 256,
                   index: index + 512]
        compensation = self.compensation(x)
        compensation = compensation[:, :, index: index + 256,
                   index: index + 512]

        sigmoid_boundary = nn.functional.sigmoid(boundary)
        sigmoid_interior = nn.functional.sigmoid(interior)

        result = boundary * sigmoid_boundary * (1 - sigmoid_interior) +\
                interior * sigmoid_interior *(1 - sigmoid_boundary) +\
                compensation * (1-sigmoid_boundary) *(1 - sigmoid_interior)

        return boundary, interior, result


class ISD(nn.Module):
    def __init__(self):
        super(ISD, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, dilation=1), nn.BatchNorm2d(128), nn.PReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2), nn.BatchNorm2d(128), nn.PReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=4, dilation=4), nn.BatchNorm2d(128), nn.PReLU(),
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1 + x)
        conv3 = self.conv3(conv2 + x)

        return conv1 + conv2 + conv3

if __name__ == '__main__':
    attention = Attention(5, 10)
    #fpn_dual = FPN_dual()


