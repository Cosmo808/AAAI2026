import logging
import pdb
from torch import nn
import torch
import copy
from models.labram import generate_labram
from einops import rearrange


class SimpleConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, feature_dim, n_subjects, dilation_factors=None):
        super(SimpleConv, self).__init__()

        dropout_ratio = 0.5
        if dilation_factors is None:
            dilation_factors = [1, 2, 4, 8, 16, 1, 2, 4, 8, 16]

        self.sequence = nn.ModuleList()
        self.sequence.append(SpatialAttention(in_channels))
        self.sequence.append(SubjectLayers(in_channels, in_channels, n_subjects))

        for i in range(num_layers):
            dilation = dilation_factors[i]
            self.sequence.append(
                nn.Sequential(
                    nn.Conv1d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
                              padding=dilation, dilation=dilation),
                    # nn.BatchNorm1d(out_channels),
                    nn.GroupNorm(num_groups=16, num_channels=out_channels),
                    nn.Dropout(dropout_ratio),
                    nn.GELU()
                )
            )

        self.final = nn.Sequential(
            nn.Conv1d(in_channels if num_layers == 0 else out_channels, 2 * out_channels, kernel_size=1, stride=1),
            nn.BatchNorm1d(2 * out_channels),
            # nn.GroupNorm(num_groups=16, num_channels=2 * out_channels),
            nn.Dropout(dropout_ratio),
            nn.GELU(),
            nn.ConvTranspose1d(2 * out_channels, feature_dim, kernel_size=1, stride=1),
        )

    def forward(self, x, subjects):
        for layer in self.sequence:
            if isinstance(layer, SubjectLayers):
                x = layer(x, subjects)
            else:
                x = layer(x)
        x = self.final(x)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels // reduction_ratio, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(channels // reduction_ratio, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn_mask = self.net(x)
        return x * attn_mask


class SubjectLayers(nn.Module):
    """Per subject linear layer."""
    def __init__(self, in_channels: int, out_channels: int, n_subjects: int, init_id: bool = False):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(n_subjects, in_channels, out_channels))
        self.biases = nn.Parameter(torch.empty(n_subjects, 1, out_channels))

        nn.init.xavier_normal_(self.weights, gain=nn.init.calculate_gain('linear'))
        nn.init.zeros_(self.biases)

        if init_id:
            assert in_channels == out_channels
            self.weights.data[:] = torch.eye(in_channels)[None]
        self.weights.data *= 1 / in_channels**0.5

    def forward(self, x, subjects):
        _, C, D = self.weights.shape
        weights = self.weights.gather(0, subjects.view(-1, 1, 1).expand(-1, C, D))
        biases = self.biases.gather(0, subjects.view(-1, 1, 1).expand(-1, 1, D))
        output = torch.einsum("bct,bcd->bdt", x, weights)
        biases = biases.permute(0, 2, 1).expand_as(output)
        output = output + biases
        return output


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, dilation_k: int):
        super().__init__()
        d1, d2 = 2 ** dilation_k, 2 ** (dilation_k + 1)

        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=d1, dilation=d1)
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=d2, dilation=d2)
        self.bn = nn.BatchNorm1d(in_channels)
        self.dropout = nn.Dropout(0.5)
        self.gelu = nn.GELU()

        self.conv3 = nn.Conv1d(in_channels, 2 * in_channels, kernel_size=3, padding=2, dilation=2)
        self.glu = nn.GLU(dim=1)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = x + residual
        x = self.conv2(x)
        x = x + residual
        x = self.bn(x)
        x = self.dropout(x)
        x = self.gelu(x)
        x = self.conv3(x)
        x = self.glu(x)
        return x


class BrainMagic(nn.Module):
    def __init__(self,
                 in_channels: int,
                 conv_channels: int,
                 out_channels: int,
                 n_subjects: int,
                 num_convblock: int = 5):
        super().__init__()

        self.sptialattention = SpatialAttention(in_channels)
        self.subject_layer = SubjectLayers(in_channels, in_channels, n_subjects)
        self.conv1x1 = nn.Conv1d(in_channels, conv_channels, kernel_size=1)

        self.blocks = nn.ModuleList()
        for k in range(1, num_convblock + 1):
            self.blocks.append(ConvBlock(conv_channels, dilation_k=k))

        self.final = nn.Sequential(
            nn.Conv1d(conv_channels, 2 * conv_channels, kernel_size=1),
            nn.Dropout(0.5),
            nn.GELU(),
            nn.Conv1d(2 * conv_channels, out_channels, kernel_size=1)
        )

    def forward(self, x, subjects):
        x = self.sptialattention(x)
        x = self.subject_layer(x, subjects)
        x = self.conv1x1(x)
        for block in self.blocks:
            x = block(x)
        return self.final(x)
