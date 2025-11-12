# -*- coding: utf-8 -*-
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch
import math
import torch.autograd as autograd
import torch.nn.functional as F
from utils.utils import polar2cartesian, cartesian2polar
import sys

# sys.path.append('../models')


# input (Tensor)
# pad (tuple)
# mode – 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
# value – fill value for 'constant' padding. Default: 0

class ConvBnRelu2d(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3, dilation=1, stride=1, groups=1, chls_group=12, is_bn=True,
                 is_relu=True, is_layer=False):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_chl, out_chl, kernel_size=kernel_size, padding=dilation * (kernel_size - 1) // 2,
                              stride=stride,
                              dilation=dilation, groups=groups, bias=True)
        self.bn = None
        self.relu = None
        self.ln = None
        if is_bn is True:
            self.bn = nn.BatchNorm2d(out_chl)
            # self.bn = nn.InstanceNorm2d(out_chl)
            # self.bn = nn.GroupNorm(num_groups=out_chl//chls_group, num_channels=out_chl)
        if is_layer is True:
            self.ln = nn.LayerNorm(out_chl)
        if is_relu is True:
            self.relu = nn.LeakyReLU(inplace=True)
            # self.relu = nn.GELU()
            # self.relu = nn.ReLU(inplace=True)
        if is_bn is True and is_layer is True:
            raise ValueError("BatchNorm adn LayerNorm can't both be True")

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.ln is not None:
            x = x.permute(0, 2, 3, 1).contiguous()
            x = self.ln(x)
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.relu is not None:
            x = self.relu(x)
        return x


class StackEncoder(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(StackEncoder, self).__init__()
        self.encode = nn.Sequential(
            ConvBnRelu2d(in_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            ConvBnRelu2d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
        )

    def forward(self, x):
        conv_out = self.encode(x)
        down_out = F.max_pool2d(conv_out, kernel_size=2, stride=2, padding=0, ceil_mode=True)

        return conv_out, down_out


class StackDecoder(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(StackDecoder, self).__init__()
        self.conv = nn.Sequential(
            ConvBnRelu2d(in_chl + out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            ConvBnRelu2d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1)
        )

    def forward(self, up_in, conv_res):
        _, _, H, W = conv_res.size()
        up_out = F.upsample(up_in, size=(H, W), mode='bilinear')
        conv_out = self.conv(torch.cat([up_out, conv_res], 1))
        return conv_out


class StackResEncoder2d(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(StackResEncoder2d, self).__init__()
        self.encode = nn.Sequential(
            ConvBnRelu2d(in_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            ConvBnRelu2d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1, is_relu=False),
        )

        self.convx = None

        if in_chl != out_chl:
            self.convx = ConvBnRelu2d(in_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1, is_bn=False,
                                      is_relu=False)

    def forward(self, x):

        if self.convx is None:
            conv_out = F.leaky_relu(self.encode(x) + x)
        else:
            conv_out = F.leaky_relu(self.encode(x) + self.convx(x))

        down_out = F.max_pool2d(conv_out, kernel_size=2, stride=2, padding=0, ceil_mode=True)

        return conv_out, down_out


class StackResCenter2d(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(StackResCenter2d, self).__init__()
        self.encode = nn.Sequential(
            ConvBnRelu2d(in_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            ConvBnRelu2d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1, is_relu=False),
        )

        self.convx = None

        if in_chl != out_chl:
            self.convx = ConvBnRelu2d(in_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1, is_bn=False,
                                      is_relu=False)

    def forward(self, x):

        if self.convx is None:
            conv_out = F.leaky_relu(self.encode(x) + x)
        else:
            conv_out = F.leaky_relu(self.encode(x) + self.convx(x))

        return conv_out


class StackResDecoder2d(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(StackResDecoder2d, self).__init__()

        self.conv1 = ConvBnRelu2d(in_chl + out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1)
        self.conv2 = ConvBnRelu2d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1,
                                  is_relu=False)
        self.convx = ConvBnRelu2d(in_chl + out_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1, is_bn=False,
                                  is_relu=False)

    def forward(self, up_in, conv_res):
        _, _, H, W = conv_res.size()
        up_out = F.upsample(up_in, size=(H, W), mode='bilinear')
        conv1 = self.conv1(torch.cat([up_out, conv_res], 1))
        conv2 = self.conv2(conv1)
        convx = F.leaky_relu(conv2 + self.convx(torch.cat([up_out, conv_res], 1)))
        return convx
    
class StackResEncoder2dNoPooling(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(StackResEncoder2dNoPooling, self).__init__()
        self.encode = nn.Sequential(
            ConvBnRelu2d(in_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            ConvBnRelu2d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1, is_relu=False),
        )

        self.convx = None

        if in_chl != out_chl:
            self.convx = ConvBnRelu2d(in_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1, is_bn=False,
                                      is_relu=False)

    def forward(self, x):

        if self.convx is None:
            conv_out = F.leaky_relu(self.encode(x) + x)
        else:
            conv_out = F.leaky_relu(self.encode(x) + self.convx(x))

        return conv_out

class ResUNetP3D(nn.Module):
    def __init__(self, in_chl=1, out_chl=1, model_chl=32):
        super(ResUNetP3D, self).__init__()
        self.out_chl = out_chl
        self.model_chl = model_chl
        self.in_chl = in_chl

        self.begin = nn.Sequential(ConvBnRelu2d(self.in_chl, self.model_chl, kernel_size=3, stride=1, is_bn=False))
        self.down1 = StackResEncoder2d(self.model_chl, self.model_chl, kernel_size=3)  # 256
        self.down2 = StackResEncoder2dNoPooling(self.model_chl * 1, self.model_chl * 2, kernel_size=3)  # 128
        self.down3 = StackResEncoder2dNoPooling(self.model_chl * 2, self.model_chl * 4, kernel_size=3)  # 64
        self.down4 = StackResEncoder2dNoPooling(self.model_chl * 4, self.model_chl * 8, kernel_size=3)  # 32

        self.center = StackResCenter2d(self.model_chl * 8, self.model_chl * 16, kernel_size=3)  # 32

        self.up4 = StackResDecoder2d(self.model_chl * 16, self.model_chl * 8, kernel_size=3)
        self.up3 = StackResDecoder2d(self.model_chl * 8, self.model_chl * 4, kernel_size=3)
        self.up2 = StackResDecoder2d(self.model_chl * 4, self.model_chl * 2, kernel_size=3)
        self.up1 = StackResDecoder2d(self.model_chl * 2, self.model_chl, kernel_size=3)

        self.end = nn.Sequential(
            ConvBnRelu2d(self.model_chl, self.out_chl, kernel_size=1, stride=1, is_bn=False, is_relu=False))

        self.begin_roi = nn.Sequential(ConvBnRelu2d(self.in_chl, self.model_chl, kernel_size=3, stride=1, is_bn=False))
        self.down1_roi = StackResEncoder2d(self.model_chl, self.model_chl, kernel_size=3)  # 256
        self.down2_roi = StackResEncoder2d(self.model_chl * 1, self.model_chl * 2, kernel_size=3)  # 128
        self.down3_roi = StackResEncoder2d(self.model_chl * 2, self.model_chl * 4, kernel_size=3)  # 64
        self.down4_roi = StackResEncoder2d(self.model_chl * 4, self.model_chl * 8, kernel_size=3)  # 32

        self.center_roi = StackResCenter2d(self.model_chl * 8, self.model_chl * 16, kernel_size=3)  # 32

        self.up4_roi = StackResDecoder2d(self.model_chl * 16, self.model_chl * 8, kernel_size=3)
        self.up3_roi = StackResDecoder2d(self.model_chl * 8, self.model_chl * 4, kernel_size=3)
        self.up2_roi = StackResDecoder2d(self.model_chl * 4, self.model_chl * 2, kernel_size=3)
        self.up1_roi = StackResDecoder2d(self.model_chl * 2, self.model_chl, kernel_size=3)

        self.end_roi = nn.Sequential(
            ConvBnRelu2d(self.model_chl, self.out_chl, kernel_size=1, stride=1, is_bn=False, is_relu=False))

        self.af_conv_e2 = nn.Sequential(
            ConvBnRelu2d(self.model_chl * 4, self.model_chl * 2, kernel_size=1, stride=1, is_bn=False, is_relu=False))
        self.af_conv_e3 = nn.Sequential(
            ConvBnRelu2d(self.model_chl * 8, self.model_chl * 4, kernel_size=1, stride=1, is_bn=False, is_relu=False))
        self.af_conv_e4 = nn.Sequential(
            ConvBnRelu2d(self.model_chl * 16, self.model_chl * 8, kernel_size=1, stride=1, is_bn=False, is_relu=False))
        self.af_center = nn.Sequential(
            ConvBnRelu2d(self.model_chl * 32, self.model_chl * 16, kernel_size=1, stride=1, is_bn=False, is_relu=False))
        self.af_conv_d4 = nn.Sequential(
            ConvBnRelu2d(self.model_chl * 16, self.model_chl * 8, kernel_size=1, stride=1, is_bn=False, is_relu=False))
        self.af_conv_d3 = nn.Sequential(
            ConvBnRelu2d(self.model_chl * 8, self.model_chl * 4, kernel_size=1, stride=1, is_bn=False, is_relu=False))
        self.af_conv_d2 = nn.Sequential(
            ConvBnRelu2d(self.model_chl * 4, self.model_chl * 2, kernel_size=1, stride=1, is_bn=False, is_relu=False))

    def forward(self, x):
        x_roi_zoomed = F.upsample(x[:, :, 192:320, 192:320], size=(512, 512), mode='bilinear', align_corners=True)
        conv0_roi = self.begin_roi(x_roi_zoomed)
        conv1_roi, d1_roi = self.down1_roi(conv0_roi)
        conv2_roi, d2_roi = self.down2_roi(d1_roi)
        conv3_roi, d3_roi = self.down3_roi(d2_roi)
        conv4_roi, d4_roi = self.down4_roi(d3_roi)
        conv5_roi = self.center_roi(d4_roi)
        up4_roi = self.up4_roi(conv5_roi, conv4_roi)
        up3_roi = self.up3_roi(up4_roi, conv3_roi)
        up2_roi = self.up2_roi(up3_roi, conv2_roi)
        up1_roi = self.up1_roi(up2_roi, conv1_roi)
        conv6_roi = self.end_roi(up1_roi)
        res_out_roi = F.leaky_relu(x_roi_zoomed + conv6_roi)

        conv0 = self.begin(x)
        conv1, d1 = self.down1(conv0)

        conv2 = self.down2(d1)
        conv2[:, :, 96:160, 96:160].data = self.af_conv_e2(torch.cat(
            [conv2[:, :, 96:160, 96:160], F.avg_pool2d(conv2_roi, kernel_size=4, stride=4, padding=0, ceil_mode=True)],
            1))
        d2 = F.max_pool2d(conv2, kernel_size=2, stride=2, padding=0, ceil_mode=True)

        conv3 = self.down3(d2)
        conv3[:, :, 32:64, 32:64].data = self.af_conv_e3(torch.cat(
            [conv3[:, :, 32:64, 32:64], F.avg_pool2d(conv3_roi, kernel_size=4, stride=4, padding=0, ceil_mode=True)],
            1))
        d3 = F.max_pool2d(conv3, kernel_size=2, stride=2, padding=0, ceil_mode=True)

        conv4 = self.down4(d3)
        conv4[:, :, 24:40, 24:40].data = self.af_conv_e4(torch.cat(
            [conv4[:, :, 24:40, 24:40], F.avg_pool2d(conv4_roi, kernel_size=4, stride=4, padding=0, ceil_mode=True)],
            1))
        d4 = F.max_pool2d(conv4, kernel_size=2, stride=2, padding=0, ceil_mode=True)

        conv5 = self.center(d4)
        conv5[:, :, 12:20, 12:20].data = self.af_center(torch.cat(
            [conv5[:, :, 12:20, 12:20], F.avg_pool2d(conv5_roi, kernel_size=4, stride=4, padding=0, ceil_mode=True)],
            1))

        up4 = self.up4(conv5, conv4)
        up4[:, :, 24:40, 24:40].data = self.af_conv_d4(torch.cat(
            [up4[:, :, 24:40, 24:40], F.avg_pool2d(up4_roi, kernel_size=4, stride=4, padding=0, ceil_mode=True)], 1))

        up3 = self.up3(up4, conv3)
        up3[:, :, 32:64, 32:64].data = self.af_conv_d3(torch.cat(
            [up3[:, :, 32:64, 32:64], F.avg_pool2d(up3_roi, kernel_size=4, stride=4, padding=0, ceil_mode=True)], 1))

        up2 = self.up2(up3, conv2)
        up2[:, :, 96:160, 96:160].data = self.af_conv_d2(torch.cat(
            [up2[:, :, 96:160, 96:160], F.avg_pool2d(up2_roi, kernel_size=4, stride=4, padding=0, ceil_mode=True)], 1))

        up1 = self.up1(up2, conv1)

        conv6 = self.end(up1)
        res_out = F.leaky_relu(x + conv6)

        return res_out, res_out_roi