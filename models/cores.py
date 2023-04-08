# Copyright © Beijing University of Posts and Telecommunications,
# School of Artificial Intelligence.


import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        return x


# Attention network
class Attentions(nn.Module):
    def __init__(self, channel_size=256):
        super(Attentions, self).__init__()
        self.A1_c = ChannelGate(channel_size)
        self.A2_c = ChannelGate(channel_size)
        self.A3_c = ChannelGate(channel_size)

    def forward(self, inputs):
        F1, F2, F3 = inputs
        # Global Average Pooling to a vector
        A1_channel = self.A1_c(F1)
        A2_channel = self.A2_c(F2)
        A3_channel = self.A3_c(F3)

        # bottom to top
        A2_channel = (A2_channel + A1_channel) / 2
        A3_channel = (A3_channel + A2_channel) / 2

        # channel pooling
        # A1 = F1 * A1_channel
        # A2 = F2 * A2_channel
        A3 = F3 * A3_channel

        return A3


# channel attention
class ChannelGate(nn.Module):
    def __init__(self, out_channels):
        super(ChannelGate, self).__init__()
        self.conv1 = nn.Conv2d(out_channels, out_channels//4, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels//4, out_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = nn.AdaptiveAvgPool2d(output_size=1)(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        # x = F.relu(self.conv1(x), inplace=True)
        # x = torch.sigmoid(self.conv2(x))
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        # self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        # if self.bn is not None:
        #     x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


# construct the top of pyramid layer
class SimpleFPA(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(SimpleFPA, self).__init__()

        self.channels_cond = in_planes
        # Master branch
        self.conv_master = BasicConv(in_planes, out_planes, kernel_size=1, stride=1)

        # Global pooling branch
        self.conv_gpb = BasicConv(in_planes, out_planes, kernel_size=1, stride=1)

    def forward(self, x):
        # Master branch
        x_master = self.conv_master(x)

        # Global pooling branch
        x_gpb = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], self.channels_cond, 1, 1)
        x_gpb = self.conv_gpb(x_gpb)

        out = x_master + x_gpb

        return out


# Feature Pyramid Network
class PyramidFeatures(nn.Module):
    def __init__(self, B1_size, B2_size, B3_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # feature_size represents the number of the channels
        self.P3_1 = SimpleFPA(B3_size, feature_size)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P2_1 = nn.Conv2d(B2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P2_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P1_1 = nn.Conv2d(B1_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P1_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        B1, B2, B3 = inputs

        P3_x = self.P3_1(B3)
        P3_upsampled_x = F.interpolate(P3_x, scale_factor=2)
        P3_x = self.P3_2(P3_x)

        P2_x = self.P2_1(B2)
        P2_x = P3_upsampled_x + P2_x
        P2_upsampled_x = F.interpolate(P2_x, scale_factor=2)
        P2_x = self.P2_2(P2_x)

        P1_x = self.P1_1(B1)
        P1_x = P1_x + P2_upsampled_x
        P1_x = self.P1_2(P1_x)

        return [P1_x, P2_x, P3_x]


class MHAM(nn.Module):
    def __init__(self, fpn_sizes, M, num_features, use_mha=True):
        super(MHAM, self).__init__()
        self.use_mha = use_mha
        if self.use_mha:
            self.fpn = PyramidFeatures(fpn_sizes[1], fpn_sizes[2], fpn_sizes[3], feature_size=M)
            # channel attention
            self.ca = Attentions(channel_size=M)
            self.conv = BasicConv(in_planes=M, out_planes=M, kernel_size=1)
        else:
            self.fpa = SimpleFPA(num_features, M)

    def forward(self, input1, input2, input3):
        if not self.use_mha:
            return self.fpa(input3)
        x = self.fpn([input1, input2, input3])
        x = self.ca(x)
        x = self.conv(x)
        return x


# bilinear polymerization pooling
class BPP(nn.Module):
    def __init__(self, epsilon):
        super(BPP, self).__init__()
        self.epsilon = epsilon

    def forward(self, features1, features2):
        # unify the size of width and height
        B, C, H, W = features1.size()
        _, M, AH, AW = features2.size()

        # match size
        if AH != H or AW != W:
            features2 = F.upsample_bilinear(features2, size=(H, W))

        # essential_matrix: (B, M, C) -> (B, M * C)
        essential_matrix = (torch.einsum('imjk,injk->imn', (features2, features1)) / float(H * W)).view(B, -1)
        # nornalize
        essential_matrix = torch.sign(essential_matrix) * torch.sqrt(torch.abs(essential_matrix) + self.epsilon)
        essential_matrix = F.normalize(essential_matrix, dim=-1)

        return essential_matrix
