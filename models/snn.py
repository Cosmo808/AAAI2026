import pdb
from torch import nn
import torch
from spikingjelly.activation_based import base, neuron, functional, surrogate, layer
import copy
from models.utils import Brain2Event
from typing import Optional, Any, Union, Callable
from timeit import default_timer as timer


class SAS(nn.Module):
    def __init__(self, args: Any):
        super().__init__()
        self.snn = SNN([args.n_channels], args.n_subjects)
        self.snn.node.step_mode = 'm'
        self.b2e = Brain2Event(args)
        self.sample_rate = args.sr

    def forward(self, data, timestamps=None, subjects=None):
        if data.ndim == 3:   # data: EEG data with size [B, C, T]
            B, C, T = data.shape
            if timestamps is None:
                timestamps = torch.arange(T) / self.sample_rate
                timestamps = timestamps.unsqueeze(0).expand(B, T).float().to(data.device)
            frames = self.b2e.forward(data, timestamps)
        if data.ndim == 4:   # frame: frame as input for SNN with size [B, T', 2, C]
            B = data.shape[0]
            frames = data

        if subjects is None:
            subjects = torch.zeros(B).to(torch.int64).to(data.device)
        vox_events = frames.permute(1, 0, 2, 3).contiguous().float()
        spikes_with_gradient = self.snn(vox_events, subjects)
        spike_idxes = [spikes_with_gradient[:, b, 0].detach().nonzero().flatten() for b in range(B)]
        return spike_idxes
        

class SNN(nn.Module):
    def __init__(self, resolution, n_subjects):
        super(SNN, self).__init__()
        self.I = []

        self.subject_layer = MultiStepSubjectLayers(
            in_channels=2,
            out_channels=2,
            n_subjects=n_subjects
        )

        assert len(resolution) == 1
        H = resolution[0]
        if H > 8:
            flatten_size = 64 * (H // 8)
            encoder_out = H // 8
        else:
            flatten_size = 64 * H
            encoder_out = H
        self.encoder = nn.Sequential(
            layer.Conv1d(2, 16, kernel_size=3, padding=1, step_mode='m'),
            layer.SeqToANNContainer(nn.GroupNorm(16, 16)),
            neuron.IFNode(step_mode='m'),
            layer.AvgPool1d(kernel_size=2, step_mode='m'),

            layer.Conv1d(16, 32, kernel_size=3, padding=1, step_mode='m'),
            layer.SeqToANNContainer(nn.GroupNorm(32, 32)),
            neuron.IFNode(step_mode='m'),
            layer.AvgPool1d(kernel_size=2, step_mode='m'),

            layer.Conv1d(32, 64, kernel_size=3, padding=1, step_mode='m'),
            layer.SeqToANNContainer(nn.GroupNorm(64, 64)),
            neuron.IFNode(step_mode='m'),
            layer.AdaptiveAvgPool1d(encoder_out, step_mode='m')
        )

        self.linear = nn.Sequential(
            layer.Flatten(step_mode='m'),
            # layer.Linear(flatten_size, 512, bias=False, step_mode='m'),
            # layer.SeqToANNContainer(nn.LayerNorm(512)),
            # neuron.IFNode(step_mode='m'),
            layer.Linear(flatten_size, 1, bias=False, step_mode='m')
        )
        self.node = IFNode()

    def forward(self, x, subjects):
        x = self.subject_layer.forward(x, subjects)
        feature_map = self.encoder(x)
        I = self.linear(feature_map)
        if self.node.step_mode == 'm':
            self.I = I
        else:
            self.I.append(I)
        x = self.node(I)
        return x

    def adjust_batch(self, idx):
        for single_layer in self.encoder:
            if isinstance(single_layer, neuron.LIFNode) or isinstance(single_layer, neuron.IFNode) or isinstance(single_layer, neuron.ParametricLIFNode):
                if not isinstance(single_layer.v, float):
                    single_layer.v = single_layer.v[idx]

        if isinstance(self.node, LIFNode) or isinstance(self.node, IFNode) or isinstance(self.node, neuron.ParametricLIFNode):
            if not isinstance(self.node.v, float):
                self.node.v = self.node.v[idx]


class IFNode(neuron.IFNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.past_v = []

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
        self.past_v.append(self.v)
        if self.v_reset is None:
            # soft reset
            self.v = self.jit_soft_reset(self.v, spike_d, self.v_threshold)
        else:
            # hard reset
            self.v = self.jit_hard_reset(self.v, spike_d, self.v_reset)

    def reset(self):
        self.past_v = []
        for key in self._memories.keys():
            self._memories[key] = copy.deepcopy(self._memories_rv[key])


class LIFNode(neuron.LIFNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.past_v = []

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
        self.past_v.append(self.v)
        if self.v_reset is None:
            # soft reset
            self.v = self.jit_soft_reset(self.v, spike_d, self.v_threshold)

        else:
            # hard reset
            self.v = self.jit_hard_reset(self.v, spike_d, self.v_reset)

    def reset(self):
        self.past_v = []
        for key in self._memories.keys():
            self._memories[key] = copy.deepcopy(self._memories_rv[key])


class SubjectLayers(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_subjects: int, init_id: bool = False):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(n_subjects, in_channels, out_channels))
        self.biases = nn.Parameter(torch.empty(n_subjects, 1, out_channels))

        nn.init.xavier_normal_(self.weights, gain=nn.init.calculate_gain('linear'))
        nn.init.zeros_(self.biases)

        if init_id:
            assert in_channels == out_channels
            self.weights.data[:] = torch.eye(in_channels)[None]
        self.weights.data *= 1 / in_channels ** 0.5

    def forward(self, x, subjects):
        # x shape: [TB, C, D]
        # subjects shape: [TB]
        weights = self.weights.gather(0, subjects.view(-1, 1, 1).expand(-1, x.shape[1], self.weights.shape[-1]))
        biases = self.biases.gather(0, subjects.view(-1, 1, 1).expand(-1, 1, self.biases.shape[-1]))
        output = torch.einsum("bct,bcd->bdt", x, weights) + biases.permute(0, 2, 1)
        return output


class MultiStepSubjectLayers(base.MultiStepModule):
    """Handles temporal dimension while preserving [T, B, C, D] format."""

    def __init__(self, in_channels: int, out_channels: int, n_subjects: int, init_id: bool = False):
        super().__init__()
        self.layer = SubjectLayers(in_channels, out_channels, n_subjects, init_id).to('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x, subjects):
        # Input: [T, B, C, D]
        T, B, C, D = x.shape
        x_flat = x.flatten(0, 1)  # [T*B, C, D]
        subjects_expanded = subjects.repeat_interleave(T)  # [T*B]
        out_flat = self.layer(x_flat, subjects_expanded)  # [T*B, C_out, D]
        return out_flat.view(T, B, -1, D)  # [T, B, C_out, D]