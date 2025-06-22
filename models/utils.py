import pdb
from timeit import default_timer as timer
import math
import torch.cuda
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import typing as tp
from pathlib import Path
from typing import Union
import soundfile as sf
import julius
import torch
import torchaudio
from scipy.interpolate import splrep, splev
from scipy.signal import resample_poly
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Model


class Brain2Event:
    def __init__(self, args):
        self.fps = args.fps
        self.C = args.C
        self.sr = args.sr

    def eeg2frame(self, data, timestamps):
        timestamps = (timestamps * self.fps).to(torch.int64) / self.fps

        events = []
        for i in range(data.shape[0]):
            data_electrode = data[i, :]
            deltas = data_electrode[1:] - data_electrode[:-1]
            deltas_abs = torch.abs(deltas)
            # threshold based on C quantile
            thre = torch.quantile(deltas_abs.float(), 1 - self.C)
            event_mask = deltas_abs >= thre
            t_indices = torch.where(event_mask)[0]
            if len(t_indices) > 0:
                event_times = timestamps[t_indices + 1].to(data.device)
                polarities = (torch.sign(deltas[t_indices]) + 1) // 2
                events.append(torch.stack([
                    torch.full((event_times.shape[0],), i, dtype=torch.int32, device=data.device),
                    event_times,
                    polarities.to(torch.int32).to(data.device)
                ], dim=1))
        events = torch.cat(events, dim=0)

        x = events[:, 0]
        t = events[:, 1].contiguous()
        p = events[:, 2]

        unique_ts = torch.unique(timestamps)
        unique_ts, _ = torch.sort(unique_ts)
        T = unique_ts.numel()
        # map event times to frame indices in compressed axis
        # searchsorted returns positions
        idx_in_frame = torch.searchsorted(unique_ts, t)

        frames = torch.zeros((T, 2, data.shape[0]), dtype=torch.int32, device=data.device)
        W, D = 2, data.shape[0]
        flat_index = idx_in_frame * (W * D) + p * D + x
        frames_flat = frames.view(-1)
        frames_flat.index_add_(0, flat_index.to(torch.int32), torch.ones_like(flat_index, dtype=torch.int32, device=data.device))
        frames = frames_flat.view(T, 2, data.shape[0])
        return frames   # [T, 2, C]

    def forward(self, data, timestamps=None):
        if data.ndim == 2:
            C, T = data.shape
            if timestamps is None:
                timestamps = torch.arange(T) / self.sr
            return self.eeg2frame(data, timestamps)
        elif data.ndim == 3:
            B, C, T = data.shape
            if timestamps is None:
                timestamps = torch.arange(T) / self.sr
                timestamps = timestamps.unsqueeze(0).expand(B, T).float().to(data.device)
            frames_batch = []
            for b in range(B):
                frames_batch.append(self.eeg2frame(data[b], timestamps[b]))
            return torch.stack(frames_batch)   # [B, T', 2, C]


class GroupReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer, group_index, mode='min', factor=0.1,
                 patience=10, verbose=False, threshold=1e-2, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8, name=None):
        """
        Args:
            group_index: Index of the parameter group to modify
            Other args: Same as ReduceLROnPlateau
        """
        # Validate inputs
        if group_index >= len(optimizer.param_groups):
            raise ValueError(f"Invalid group_index {group_index} for optimizer with {len(optimizer.param_groups)} groups")

        super().__init__(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode=threshold_mode,
            eps=eps,
        )

        # Override group-specific parameters
        self.group_index = group_index
        self.group = self.optimizer.param_groups[group_index]
        self.min_lr = min_lr
        self.name = name or f"group_{group_index}"
        self.cooldown = cooldown
        self.verbose = verbose
        self._reset()

    def _reset(self):
        """Reset all scheduler state variables"""
        self.best = 1e8 if self.mode == 'min' else -1e8
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self.last_epoch = 0  # Tracks total steps called

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def step(self, metrics):
        current = metrics
        self.last_epoch += 1  # Increment at start (matches PyTorch convention)

        if self.in_cooldown:
            self.cooldown_counter -= 1
            # if self.verbose:
            #     print(f"Cooldown counter: {self.cooldown_counter}")
            return

        # Initialize best on first step
        if self.best is None:
            self.best = current
            return

        # Check if current metric is better
        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        # Reduce LR if no improvement for 'patience' epochs
        if self.num_bad_epochs > self.patience:
            self._reduce_lr(self.last_epoch)
            self.cooldown_counter = self.cooldown  # Start cooldown
            self.num_bad_epochs = 0  # Reset counter

    def _reduce_lr(self, epoch):
        old_lr = self.group['lr']
        new_lr = max(old_lr * self.factor, self.min_lr)

        if abs(old_lr - new_lr) > self.eps:
            self.group['lr'] = new_lr
            if self.verbose:
                print(f"Reducing {self.name} LR from {old_lr:.2e} to {new_lr:.2e}")

    def is_better(self, a, best):
        if self.mode == 'min':
            if self.threshold_mode == 'rel':
                return a < best - best * self.threshold
            else:
                return a < best - self.threshold
        else:
            if self.threshold_mode == 'rel':
                return a > best + best * self.threshold
            else:
                return a > best + self.threshold


class GroupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, group_index, T_max, eta_min=0, last_epoch=-1, verbose=False, name=None):
        """
        Args:
            optimizer: Wrapped optimizer with multiple parameter groups.
            group_index: Index of the parameter group to apply this scheduler to.
            T_max: Maximum number of iterations.
            eta_min: Minimum learning rate.
            last_epoch: The index of last epoch. Default: -1.
            verbose: If True, prints a message to stdout for each update.
            name: Optional name for logging.
        """
        if group_index >= len(optimizer.param_groups):
            raise ValueError(f"Invalid group_index {group_index} for optimizer with {len(optimizer.param_groups)} groups")

        self.group_index = group_index
        self.group = optimizer.param_groups[group_index]
        self.T_max = T_max
        self.eta_min = eta_min
        self.verbose = verbose
        self.name = name or f"group_{group_index}"
        self.base_lr = self.group['lr']

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return [self.base_lr]

        cos_inner = math.pi * self.last_epoch / self.T_max
        new_lr = self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(cos_inner)) / 2
        return [new_lr]

    def step(self, epoch=None):
        """Performs a single scheduler step and updates only the group-specific LR"""
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

        new_lr = self.get_lr()[0]
        self.group['lr'] = new_lr

        if self.verbose:
            print(f"GroupCosineAnnealingLR ({self.name}): Updated LR to {new_lr:.2e}")


class wav_processor:
    def __init__(self, model_name="facebook/wav2vec2-base-10k-voxpopuli"):
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model.eval()

    def resample(self, rep, sample_num):
        if rep.shape[-1] == sample_num:
            return rep
        elif rep.shape[-1] < sample_num:  # upsample
            if rep.dim() == 3:
                batch, features, time = rep.shape
                x_original = np.linspace(0, 1, time)
                x_target = np.linspace(0, 1, sample_num)
                interpolated_rep = np.zeros((batch, features, sample_num))
                for i in range(batch):
                    for j in range(features):
                        tck = splrep(x_original, rep[i, j, :], k=3, s=0)
                        interpolated_rep[i, j, :] = splev(x_target, tck)
            elif rep.dim() == 2:
                features, time = rep.shape
                x_original = np.linspace(0, 1, time)
                x_target = np.linspace(0, 1, sample_num)
                interpolated_rep = np.zeros((features, sample_num))
                for j in range(features):
                    tck = splrep(x_original, rep[j, :].cpu().numpy(), k=3, s=0)
                    interpolated_rep[j, :] = splev(x_target, tck)
            return torch.tensor(interpolated_rep)
        elif rep.shape[-1] > sample_num:  # downsample
            rep = resample_poly(rep, up=sample_num, down=rep.shape[-1], axis=1)
            return torch.tensor(rep)

    def extract_wav(self, filepath, onset: float, offset: float):
        try:
            info = torchaudio.info(filepath)
            sr = float(info.sample_rate)
        except RuntimeError:
            with sf.SoundFile(filepath) as f:
                sr = float(f.samplerate)
        frame_offset = np.round(onset * sr).astype(int) if isinstance(onset, np.ndarray) else int(round(onset * sr))
        num_frames = np.round((offset - onset) * sr).astype(int) if isinstance((offset - onset), np.ndarray) else int(
            round((offset - onset) * sr))
        wav = torchaudio.load(filepath, frame_offset=frame_offset, num_frames=num_frames)[0]
        delta = abs(wav.shape[-1] / sr - offset + onset)
        assert delta < 1e-5, (delta, filepath, onset, offset, onset - offset)
        return wav, sr

    def wav2vec(self, sound_event, start: float, stop: float):
        sound_start = np.array(sound_event['start'].tolist())
        index = (sound_start > start).argmax()
        index -= 1
        filepath = (sound_event.iloc[index])['filepath']
        start -= sound_start[index]
        stop -= sound_start[index]

        try:
            wav, sr = self.extract_wav(filepath, start, stop)
        except AssertionError:
            return None, None
        wav = torch.mean(wav, dim=0)  # stereo to mono

        model_sr = self.feature_extractor.sampling_rate
        wav = julius.resample.ResampleFrac(old_sr=int(sr), new_sr=model_sr)(wav)

        # [1, T]
        processed_wav = self.feature_extractor(wav, return_tensors="pt", sampling_rate=model_sr, do_normalize=True).input_values
        with torch.no_grad():
            outputs = self.model(processed_wav, output_hidden_states=True)
        hidden_states = outputs.get("hidden_states")
        last_hidden_state = outputs.get("last_hidden_state")
        if isinstance(hidden_states, tuple):
            hidden_states = torch.stack(hidden_states)
        # hidden_states[0] is equal to last_hidden_state
        return hidden_states, last_hidden_state


class RidgeRegression(nn.Module):
    def __init__(self, input_dim, output_dim=1, alpha=1.0, device=None):
        super(RidgeRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        self.alpha = alpha  # L2 regularization strength
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        return self.linear(x)

    def train_model(self, X, y, epochs=1000, lr=1e-3, verbose=False):
        X = X.to(self.device)
        y = y.to(self.device)

        optimizer = optim.SGD(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()

            outputs = self(X)
            loss = criterion(outputs, y)

            # Ridge (L2) regularization term: alpha * ||W||^2
            l2_reg = self.alpha * torch.norm(self.linear.weight, p=2) ** 2
            total_loss = loss + l2_reg

            total_loss.backward()
            optimizer.step()

            if verbose and (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss.item():.4f}')

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            return self(X.to(self.device))


class ConvLinear(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.conv = nn.Conv1d(dim1, dim1 // 8, kernel_size=3, stride=1, padding=1, dilation=1)
        self.linear = nn.Linear(dim1 // 8 * dim2, dim1)
        self.to(self.device)

    def forward(self, x):  # x: [B, 768, input_dim]
        B, dim1, dim2 = x.shape
        x = self.conv(x)  # [B, 769//8, 600]
        x = x.view(B, dim1 // 8 * dim2)  # [B, 769//8 * 600]
        x = self.linear(x)  # [B, 768]
        return x

    def train_model(self, X, y, epochs=1000, lr=1e-3, verbose=False):
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=25, shuffle=True)

        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0

            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)

                optimizer.zero_grad()
                out = self(xb)
                loss = criterion(out, yb)

                loss.backward()
                optimizer.step()

                total_loss += loss.detach().item()

            if verbose and (epoch + 1) % 50 == 0:
                avg_loss = total_loss / len(loader)
                print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            return self.forward(X.to(self.device))