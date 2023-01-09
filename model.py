import math
import torch
from torch import nn
import torchvision


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, groups):
        super(CausalConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.pad, dilation=dilation, groups=groups)

    def forward(self, x):
        x = self.conv1(x)
        x = x[..., :-self.pad]
        return x


class WaveNetResidualBlock(nn.Module):
    def __init__(self, n_channel, n_mul, frames, kernel_size, dilation_rate, n_groups=1):
        super(WaveNetResidualBlock, self).__init__()
        self.sigmoid_conv = nn.Sequential(
            nn.LayerNorm([n_channel * n_mul, frames]),
            CausalConv1d(n_channel * n_mul, n_channel * n_mul, kernel_size, dilation_rate, n_groups),
            nn.Sigmoid()
        )
        self.tanh_conv = nn.Sequential(
            nn.LayerNorm([n_channel * n_mul, frames]),
            CausalConv1d(n_channel * n_mul, n_channel * n_mul, kernel_size, dilation_rate, n_groups),
            nn.Tanh()
        )
        self.skip_connection = nn.Sequential(
            nn.LayerNorm([n_channel * n_mul, frames]),
            nn.Conv1d(n_channel * n_mul, n_channel, 1, groups=n_groups)
        )
        self.residual = nn.Sequential(
            nn.LayerNorm([n_channel * n_mul, frames]),
            nn.Conv1d(n_channel * n_mul, n_channel * n_mul, 1, groups=n_groups)
        )

    def forward(self, x):
        sigmoid_conv = self.sigmoid_conv(x)
        tanh_conv = self.tanh_conv(x)
        mul = torch.mul(sigmoid_conv, tanh_conv)
        skip = self.skip_connection(mul)
        residual = self.residual(mul)
        return skip, residual + x


class WaveNet(nn.Module):
    def __init__(self, n_blocks, n_channel, n_mul, frames, kernel_size, n_groups):
        super(WaveNet, self).__init__()
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size
        self.feature_layer = nn.Sequential(
            nn.LayerNorm([n_channel, frames]),
            nn.Conv1d(n_channel, n_channel * n_mul, 1, groups=n_groups)
        )
        self.blocks = nn.ModuleList([WaveNetResidualBlock(n_channel, n_mul, frames, kernel_size, 2 ** i, n_groups) for i in range(n_blocks)])
        self.skip_connection = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm([n_channel, frames]),
            nn.Conv1d(n_channel, n_channel, 1, groups=n_groups),
            nn.ReLU(),
            nn.LayerNorm([n_channel, frames]),
            nn.Conv1d(n_channel, n_channel, 1, groups=n_groups)
        )

    def get_receptive_field(self):
        rf = 1
        for _ in range(self.n_blocks):
            rf = rf * 2 + self.kernel_size - 2
        return rf

    def forward(self, x):
        x = self.feature_layer(x)
        skips = []
        for idx, block in enumerate(self.blocks):
            skip, x = block(x)
            skips.append(skip)
        skips = torch.stack(skips).sum(0)
        output = self.skip_connection(skips)
        return output[..., self.get_receptive_field() - 1:-1]


class ResNet(nn.Module):
    def __init__(self, n_class):
        super(ResNet, self).__init__()
        self.resnet = torchvision.models.resnet50()
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = nn.Linear(in_features=2048, out_features=n_class, bias=True)

    def forward(self, x):
        output = self.resnet(x)
        return output


class MTLClass(nn.Module):
    def __init__(self, n_blocks, n_channel, n_mul, frames, kernel_size, n_groups, n_class, arcface=None):
        super(MTLClass, self).__init__()
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size
        self.feature_layer = nn.Sequential(
            nn.LayerNorm([n_channel, frames]),
            nn.Conv1d(n_channel, n_channel * n_mul, 1, groups=n_groups)
        )
        self.blocks = nn.ModuleList([WaveNetResidualBlock(n_channel, n_mul, frames, kernel_size, 2 ** i, n_groups) for i in range(n_blocks)])
        self.recon_layer = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm([n_channel, frames]),
            nn.Conv1d(n_channel, n_channel, 1, groups=n_groups),
            nn.ReLU(),
            nn.LayerNorm([n_channel, frames]),
            nn.Conv1d(n_channel, n_channel, 1, groups=n_groups)
        )
        self.class_layer = torchvision.models.resnet18()
        self.class_layer.conv1 = nn.Conv2d(n_blocks, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.class_layer.fc = nn.Linear(in_features=512, out_features=n_class, bias=True)

    def get_receptive_field(self):
        rf = 1
        for _ in range(self.n_blocks):
            rf = rf * 2 + self.kernel_size - 2
        return rf

    def forward(self, x):
        x = self.feature_layer(x)
        skips = []
        for idx, block in enumerate(self.blocks):
            skip, x = block(x)
            skips.append(skip)
        skips = torch.stack(skips, dim=1)
        output1 = self.recon_layer(skips.sum(1))
        output2 = self.class_layer(skips)
        return output1[..., self.get_receptive_field() - 1:-1], output2


class MultiResolutionWaveNet(nn.Module):
    def __init__(self, n_blocks, n_channel, n_mul, frames, kernel_size, n_groups):
        super(MultiResolutionWaveNet, self).__init__()
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size
        self.feature_layer = nn.Sequential(
            nn.LayerNorm([n_channel, frames]),
            nn.Conv1d(n_channel, n_channel * n_mul, 1, groups=n_groups)
        )
        self.blocks = nn.ModuleList([WaveNetResidualBlock(n_channel, n_mul, frames, kernel_size, 2 ** i, n_groups) for i in range(n_blocks)])
        self.recon_layers = nn.ModuleList([nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm([n_channel, frames]),
            nn.Conv1d(n_channel, n_channel, 1, groups=n_groups),
            nn.ReLU(),
            nn.LayerNorm([n_channel, frames]),
            nn.Conv1d(n_channel, n_channel, 1, groups=n_groups)
        ) for _ in range(n_blocks)])

    def get_receptive_field(self):
        rf = 1
        for _ in range(self.n_blocks):
            rf = rf * 2 + self.kernel_size - 2
        return rf

    def forward(self, x):
        x = self.feature_layer(x)
        skips = []
        for idx, (block, recon) in enumerate(zip(self.blocks, self.recon_layers)):
            skip, x = block(x)
            skip_recon = recon(skip)
            skips.append(skip_recon)
        skips = torch.stack(skips, dim=1)
        return skips[..., self.get_receptive_field() - 1:-1]


class MultiResolutionSumWaveNet(nn.Module):
    def __init__(self, n_blocks, n_channel, n_mul, frames, kernel_size, n_groups):
        super(MultiResolutionSumWaveNet, self).__init__()
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size
        self.feature_layer = nn.Sequential(
            nn.LayerNorm([n_channel, frames]),
            nn.Conv1d(n_channel, n_channel * n_mul, 1, groups=n_groups)
        )
        self.blocks = nn.ModuleList([WaveNetResidualBlock(n_channel, n_mul, frames, kernel_size, 2 ** i, n_groups) for i in range(n_blocks)])
        self.recon_layers = nn.ModuleList([nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm([n_channel, frames]),
            nn.Conv1d(n_channel, n_channel, 1, groups=n_groups),
            nn.ReLU(),
            nn.LayerNorm([n_channel, frames]),
            nn.Conv1d(n_channel, n_channel, 1, groups=n_groups)
        ) for _ in range(n_blocks)])

    def get_receptive_field(self):
        rf = 1
        for _ in range(self.n_blocks):
            rf = rf * 2 + self.kernel_size - 2
        return rf

    def forward(self, x):
        x = self.feature_layer(x)
        skips = []
        skip = 0
        for idx, (block, recon) in enumerate(zip(self.blocks, self.recon_layers)):
            skip_layer, x = block(x)
            skip += skip_layer
            skip_recon = recon(skip)
            skips.append(skip_recon)
        skips = torch.stack(skips, dim=1)
        return skips[..., self.get_receptive_field() - 1:-1]


class MTLSeg(nn.Module):
    def __init__(self, n_blocks, n_channel, n_mul, frames, kernel_size, n_groups, n_class):
        super(MTLSeg, self).__init__()
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size
        self.feature_layer = nn.Sequential(
            nn.LayerNorm([n_channel, frames]),
            nn.Conv1d(n_channel, n_channel * n_mul, 1, groups=n_groups)
        )
        self.blocks = nn.ModuleList([WaveNetResidualBlock(n_channel, n_mul, frames, kernel_size, 2 ** i, n_groups) for i in range(n_blocks)])
        self.reconstruct = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm([n_channel, frames]),
            nn.Conv1d(n_channel, n_channel, 1, groups=n_groups),
            nn.ReLU(),
            nn.LayerNorm([n_channel, frames]),
            nn.Conv1d(n_channel, n_channel, 1, groups=n_groups)
        )
        self.segment = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm([n_channel, frames]),
            nn.Conv1d(n_channel, n_channel, 1, groups=n_groups),
            nn.ReLU(),
            nn.LayerNorm([n_channel, frames]),
            nn.Conv1d(n_channel, n_class, 1)
        )

    def get_receptive_field(self):
        rf = 1
        for _ in range(self.n_blocks):
            rf = rf * 2 + self.kernel_size - 2
        return rf

    def forward(self, x):
        x = self.feature_layer(x)
        skips = []
        for block in self.blocks:
            skip, x = block(x)
            skips.append(skip)
        skips = torch.stack(skips).sum(0)
        output1 = self.reconstruct(skips)
        output2 = self.segment(skips)
        return output1[..., self.get_receptive_field() - 1:-1], output2[..., self.get_receptive_field() - 1:-1]


class MTLClassSeg(nn.Module):
    def __init__(self, n_blocks, n_channel, n_mul, frames, kernel_size, n_groups, n_class, arcface=None):
        super(MTLClassSeg, self).__init__()
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size
        self.feature_layer = nn.Sequential(
            nn.LayerNorm([n_channel, frames]),
            nn.Conv1d(n_channel, n_channel * n_mul, 1, groups=n_groups)
        )
        self.blocks = nn.ModuleList([WaveNetResidualBlock(n_channel, n_mul, frames, kernel_size, 2 ** i, n_groups) for i in range(n_blocks)])
        self.reconstruction = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm([n_channel, frames]),
            nn.Conv1d(n_channel, n_channel, 1, groups=n_groups),
            nn.ReLU(),
            nn.LayerNorm([n_channel, frames]),
            nn.Conv1d(n_channel, n_channel, 1, groups=n_groups)
        )

        self.classification = torchvision.models.resnet18()
        self.classification.conv1 = nn.Conv2d(n_blocks, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.classification.fc = nn.Linear(in_features=512, out_features=n_class, bias=True)

        self.segmentation = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm([n_channel, frames]),
            nn.Conv1d(n_channel, n_channel, 1, groups=n_groups),
            nn.ReLU(),
            nn.LayerNorm([n_channel, frames]),
            nn.Conv1d(n_channel, n_class, 1)
        )

    def get_receptive_field(self):
        rf = 1
        for _ in range(self.n_blocks):
            rf = rf * 2 + self.kernel_size - 2
        return rf

    def forward(self, x):
        x = self.feature_layer(x)
        skips = []
        for idx, block in enumerate(self.blocks):
            skip, x = block(x)
            skips.append(skip)
        skips = torch.stack(skips, dim=1)
        output1 = self.reconstruction(skips.sum(1))
        output2 = self.classification(skips)
        output3 = self.segmentation(skips.sum(1))
        return output1[..., self.get_receptive_field() - 1:-1], output2, output3[..., self.get_receptive_field() - 1:-1]


