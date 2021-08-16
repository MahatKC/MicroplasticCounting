""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            #nn.ReLU(inplace=True)
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, channel_resize, conv_size, num_skip_connections, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(channel_resize * 2, channel_resize, kernel_size=2, stride=2)
            self.conv = DoubleConv(conv_size, out_channels)

        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.resize1 = nn.Conv2d(64, channel_resize, kernel_size=1)
        self.channel_resize = []
        self.channel_resize.append(self.resize1)
        if num_skip_connections >= 2:
            self.resize2 = nn.Conv2d(128, channel_resize, kernel_size=1)
            self.channel_resize.append(self.resize2)
        if num_skip_connections >= 3:
            self.resize3 = nn.Conv2d(256, channel_resize, kernel_size=1)
            self.channel_resize.append(self.resize3)
        if num_skip_connections >= 4:
            self.resize4 = nn.Conv2d(512, channel_resize, kernel_size=1)
            self.channel_resize.append(self.resize4)

    def forward(self, x, down_weights):

        x = self.up(x)
        # # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]

        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        target_shape = tuple(x.shape[2:4])
        for i, d in enumerate(down_weights):
            d = F.interpolate(d, size=target_shape, mode='bicubic', align_corners=True)
            d = self.channel_resize[i](d)
            d = self.relu(d)
            x = torch.cat([x, d], dim=1)

        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
