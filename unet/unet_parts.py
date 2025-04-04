""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


# Define Swish activation
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class DoubleConv3(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConv5(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=5, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)

class TripleConv3(nn.Module):
    """(convolution => [BN] => ReLU) * 3"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.triple_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.triple_conv(x)

class TripleConv5(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.triple_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=5, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=5, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.triple_conv(x)


class DownDouble(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        if kernel_size == 5:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv5(in_channels, out_channels)
            )
        else:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv3(in_channels, out_channels)
            )

    def forward(self, x):
        return self.maxpool_conv(x)

class DownTriple(nn.Module):
    """Downscaling with maxpool then triple conv"""

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        if kernel_size == 5:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                TripleConv5(in_channels, out_channels)
            )
        else:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                TripleConv3(in_channels, out_channels)
            )  

    def forward(self, x):
        return self.maxpool_conv(x)

class UpTriple(nn.Module):
    """Triple conv then upscaling"""

    def __init__(self, in_channels, out_channels, kernel_size, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if kernel_size == 5:
            if bilinear:
                self.conv = TripleConv5(in_channels, out_channels // 2)
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            else:
                self.conv = TripleConv5(in_channels, out_channels)
                self.up = nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=2, stride=2)
        else:
            if bilinear:
                self.conv = TripleConv3(in_channels, out_channels // 2)
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            else:
                self.conv = TripleConv3(in_channels, out_channels)
                self.up = nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return x

class UpDouble(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if kernel_size == 5:
            if bilinear:
                self.conv = DoubleConv5(in_channels, out_channels // 2)
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            else:
                self.conv = DoubleConv5(in_channels, out_channels)
                self.up = nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=2, stride=2)
        else:
            if bilinear:
                self.conv = DoubleConv3(in_channels, out_channels // 2)
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            else:
                self.conv = DoubleConv3(in_channels, out_channels)
                self.up = nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = DoubleConv3(in_channels, out_channels)
        self.conv_out = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv(x)
        return self.conv_out(x1)
