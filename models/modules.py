import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class Conv3dPad(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding_mode="zeros", groups=1, bias=False):
        super().__init__()
        assert padding_mode in ['zeros', 'circular']
        self.padding_mode = padding_mode
        if isinstance(kernel_size, int):
            self.padding = [(kernel_size - 1) // 2, (kernel_size - 1) // 2, (kernel_size - 1) // 2] 
        else:
            self.padding = [(k - 1) // 2 for k in kernel_size]
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=0, groups=groups, bias=bias)

    def forward(self, x):
        # 3D padding: (angel_1, angle_2, subcarrier)
        # Only apply circular padding in angle domain
        if self.padding_mode == 'circular':
            x = F.pad(x, (0, 0, self.padding[1], self.padding[1],
                          self.padding[0], self.padding[0]), mode='circular')
            x = F.pad(x, (self.padding[2], self.padding[2], 0, 0,
                          0, 0), mode='constant', value=0)
        else:
            x = F.pad(x, (self.padding[2], self.padding[2], self.padding[1], self.padding[1],
                          self.padding[0], self.padding[0]), mode='constant', value=0)
        return self.conv(x)


class ConvBN3d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding_mode='zeros', groups=1, bias=False, factorization=False):
        modules = []
        if not factorization:
            modules.append(("conv", Conv3dPad(in_channels, out_channels, kernel_size, stride, padding_mode, groups, bias)))
        else:
            # conv factorization
            modules.append(("conv_dh", Conv3dPad(in_channels, out_channels, (kernel_size, kernel_size, 1), 1, padding_mode, groups, bias)))
            modules.append(("conv_w", Conv3dPad(out_channels, out_channels, (1, 1, kernel_size), stride, padding_mode, groups, bias)))

        modules.append(("bn", nn.BatchNorm3d(out_channels)))
        modules.append(("act", nn.SiLU()))
        super().__init__(OrderedDict(modules))


class MultiConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_in, kernel_1, kernel_2, padding_mode='zeros', shortcut=True, groups=1, factorization=False):
        super().__init__()
        self.in_conv = ConvBN3d(in_channels, out_channels, kernel_in, 1, padding_mode, groups, factorization)
        self.conv1 = ConvBN3d(out_channels, out_channels, kernel_1, 1, padding_mode, groups, factorization)
        self.conv2 = ConvBN3d(out_channels, out_channels, kernel_2, 1, padding_mode, groups, factorization)
        self.add = shortcut
        self.conv1d = ConvBN3d(2 * out_channels, out_channels, 1, 1, padding_mode="zeros", groups=groups)
        self.act = nn.SiLU()

    def forward(self, x):
        out = self.in_conv(x)
        cat = torch.cat((self.conv1(out), self.conv2(out)), dim=1)
        out = self.conv1d(cat)
        return self.act(out + self.in_conv(x)) if self.add else self.act(out)


class UpCatConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding_mode='zeros', groups=1, factorization=False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv = nn.Sequential(OrderedDict([
            ("conv1", ConvBN3d(in_channels * 2, in_channels, kernel_size, stride, padding_mode, groups, factorization)),
            ("conv2", ConvBN3d(in_channels, out_channels, kernel_size, stride, padding_mode, groups, factorization))
        ]))

    def forward(self, x1, x2):
        return self.conv(torch.cat((self.up(x1), x2), dim=1))


class DownCatConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding_mode='zeros', groups=1, factorization=False):
        super().__init__()
        self.downconv = ConvBN3d(in_channels, in_channels, kernel_size, stride=2, padding_mode=padding_mode, groups=groups, factorization=factorization)
        self.conv = ConvBN3d(in_channels * 2, out_channels, kernel_size, stride, padding_mode=padding_mode, groups=groups, factorization=factorization)

    def forward(self, x1, x2):
        return self.conv(torch.cat((self.downconv(x1), x2), dim=1))


class HeadConv3d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding_mode='zeros', groups=1, factorization=False):
        layer = []
        if not factorization:
            layer.append(("conv", Conv3dPad(in_channels, out_channels, kernel_size, stride, padding_mode, groups, bias=False)))
        else:
            layer.append(("conv1", Conv3dPad(in_channels, out_channels, (kernel_size, kernel_size, 1), stride, padding_mode, groups, bias=False)))
            layer.append(("conv2", Conv3dPad(out_channels, out_channels, (1, 1, kernel_size), stride, padding_mode, groups, bias=False)))
        layer.append(("bn", nn.BatchNorm3d(out_channels)))
        layer.append(("act", nn.LeakyReLU(negative_slope=0.3, inplace=True)))
        super().__init__(OrderedDict(layer))
