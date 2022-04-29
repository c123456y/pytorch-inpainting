import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layer import GatedConv2d
from utilatten.attention import eca_layer


class UpConv(nn.Module):
    def __init__(self, inc, outc, scale=2):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2d(inc, outc, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True))


class DUC(nn.Module):
    def __init__(self, inchannel, outchannel, upscale_factor=2):
        super(DUC, self).__init__()
        self.conv = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1)
        self.gn = nn.GroupNorm(num_groups=32, num_channels=outchannel)
        self.relu = nn.ReLU()
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x


class MiBlock(nn.Module):
    def __init__(self, dim):
        super(MiBlock, self).__init__()

        self.f1 = nn.Sequential(
            nn.Conv2d(dim, dim//8, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(2),
            GatedConv2d(dim//8, dim // 4, kernel_size=3, stride=1, padding=0, dilation=2, groups=1, batch_norm=True,  # [b,64,64,64]
                   activation=nn.ReLU(inplace=True))
        )
        self.f2 = nn.Sequential(
            nn.Conv2d(dim, dim//8, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3),
            GatedConv2d(dim//8, dim // 4, kernel_size=3, stride=1, padding=0, dilation=3, groups=1, batch_norm=True,  # [b,64,64,64]
                    activation=nn.ReLU(inplace=True))
        )
        self.f3 = nn.Sequential(
            nn.Conv2d(dim, dim//8, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(5),
            GatedConv2d(dim//8, dim // 4, kernel_size=3, stride=1, padding=0, dilation=5, groups=1, batch_norm=True,  # [b,64,64,64]
                        activation=nn.ReLU(inplace=True))
        )
        self.f4 = nn.Sequential(
            nn.Conv2d(dim, dim//8, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(7),
            GatedConv2d(dim//8, dim // 4, kernel_size=3, stride=1, padding=0, dilation=7, groups=1, batch_norm=True,  # [b,64,64,64]
                        activation=nn.ReLU(inplace=True))
        )
        self.fuse = nn.Sequential(
            nn.ReflectionPad2d(1),
            GatedConv2d(dim, dim, kernel_size=3, stride=1, padding=0, dilation=1, batch_norm=True,
                        activation=None)
            )
        self.eca = eca_layer(dim, k_size=3)

        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))

    def forward(self, x):
        out1 = self.f1(x)
        out2 = self.f2(x)
        out3 = self.f3(x)
        out4 = self.f4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)

        out = self.fuse(out)
        out = self.eca(out)

        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + out * mask


def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat



