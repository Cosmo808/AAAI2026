import os
import pdb
from tqdm import tqdm
import re
import mne
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import torch
import torchaudio
import soundfile as sf
import julius
import argparse
from scipy.interpolate import splrep, splev
from scipy.signal import resample_poly
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Model


def eeg2frame(data, timestamps, C):
    # timestamps: 1D int tensor of discrete frame indices (possibly non-contiguous)
    events = []
    for i in range(data.shape[0]):
        data_electrode = data[i, :]
        deltas = data_electrode[1:] - data_electrode[:-1]
        deltas_abs = torch.abs(deltas)
        # threshold based on C quantile
        thre = torch.quantile(deltas_abs.float(), 1 - C)
        event_mask = deltas_abs >= thre
        t_indices = torch.where(event_mask)[0]
        if len(t_indices) > 0:
            event_times = timestamps[t_indices + 1]
            polarities = (torch.sign(deltas[t_indices]) + 1) // 2
            events.append(torch.stack([
                torch.full((event_times.shape[0],), i, dtype=torch.int32),
                event_times,
                polarities.to(torch.int32)
            ], dim=1))
    events = torch.cat(events, dim=0)

    x = events[:, 0]
    t = events[:, 1].contiguous()
    p = events[:, 2]
    # build unique timestamp axis
    unique_ts = torch.unique(timestamps)
    unique_ts, _ = torch.sort(unique_ts)
    T = unique_ts.numel()
    # map event times to frame indices in compressed axis
    # searchsorted returns positions
    idx_in_frame = torch.searchsorted(unique_ts, t)

    frames = torch.zeros((T, 2, data.shape[0]), dtype=torch.int32)
    W, D = 2, data.shape[0]
    flat_index = idx_in_frame * (W * D) + p * D + x
    frames_flat = frames.view(-1)
    frames_flat.index_add_(0, flat_index, torch.ones_like(flat_index, dtype=torch.int32))
    frames = frames_flat.view(T, 2, data.shape[0])
    return frames


def process_subject_event(base_path, idx, run, args):
    random.seed(int(idx) + int(run))
    # obtain file path
    files = os.listdir(os.path.join(base_path, f"{idx}_run{run}"))
    for file in files:
        if file.endswith('.fif'):
            fif_file = file
        elif file.endswith('.csv'):
            event_file = file
    fif_path = os.path.join(base_path, f"{idx}_run{run}", fif_file)
    event_path = os.path.join(base_path, f"{idx}_run{run}", event_file)

    # load eeg and event data
    sample_rate = re.findall(r'sr(\d+)-', fif_file)
    sample_rate = int(sample_rate[0]) if sample_rate else 0
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=0)
    data, times = raw[:, :]
    data = torch.tensor(data).float().cpu()
    times = torch.tensor(times).float().cpu()
    events = pd.read_csv(event_path)
    sound_events = events[events['kind'] == 'sound']
    block_events = events[events['kind'] == 'block']
    word_events = events[events['kind'] == 'word']
    sound_events.reset_index(drop=True, inplace=True)
    block_events.reset_index(drop=True, inplace=True)
    word_events.reset_index(drop=True, inplace=True)

    timestamp = block_events['start'].tolist()
    duration = block_events['duration'].tolist()
    duration[-1] = word_events['start'].tolist()[-1] + word_events['duration'].tolist()[-1] - timestamp[-1]

    # random split on blocks (4:1)
    num_blocks = len(timestamp)
    test_num = int(args.p * num_blocks)
    block_indices = list(range(num_blocks))
    test_blocks = sorted(random.sample(block_indices, test_num))
    train_blocks = sorted(set(block_indices) - set(test_blocks))

    segments_test = []
    segments_train = []
    for b in test_blocks:
        s, e = int(timestamp[b] * sample_rate), int((timestamp[b] + duration[b]) * sample_rate)
        segments_test.append((s, e))
    for b in train_blocks:
        s, e = int(timestamp[b] * sample_rate), int((timestamp[b] + duration[b]) * sample_rate)
        segments_train.append((s, e))

    segments_test.sort(key=lambda x: x[0])
    segments_train.sort(key=lambda x: x[0])

    data_test = torch.cat([data[:, s:e] for s, e in segments_test], dim=1)
    data_train = torch.cat([data[:, s:e] for s, e in segments_train], dim=1)
    time_test = torch.cat([times[s:e] for s, e in segments_test], dim=0)
    time_train = torch.cat([times[s:e] for s, e in segments_train], dim=0)

    # eeg to event to frame
    fps = args.fps
    time_test_idx = (time_test * fps).to(torch.int32)
    time_train_idx = (time_train * fps).to(torch.int32)

    frames_test = eeg2frame(data_test, time_test_idx, args.C)
    frames_train = eeg2frame(data_train, time_train_idx, args.C)

    # frame corresponds to timestamp
    # use original discrete timestamps
    unique_test_ts = torch.unique(time_test_idx)
    unique_test_ts, _ = torch.sort(unique_test_ts)
    time_test = unique_test_ts.to(torch.float32) / fps
    unique_train_ts = torch.unique(time_train_idx)
    unique_train_ts, _ = torch.sort(unique_train_ts)
    time_train = unique_train_ts.to(torch.float32) / fps

    # block start and end times (discrete, sorted)
    start_end_test = []
    for b in test_blocks:
        start_end_test += [timestamp[b], timestamp[b] + duration[b]]
    start_end_train = []
    for b in train_blocks:
        start_end_train += [timestamp[b], timestamp[b] + duration[b]]
    sp_test = sorted([int(t * fps) / fps for t in start_end_test])
    sp_train = sorted([int(t * fps) / fps for t in start_end_train])
    start_time_test = torch.tensor(sp_test)
    start_time_train = torch.tensor(sp_train)

    ##################################################################################

    # # continuous split train test in block
    # test_num = int(args.p * len(timestamp))
    # test_start = random.randint(0, len(timestamp) - test_num - 1)
    # test_end = test_start + test_num
    #
    # start_time_test = timestamp[test_start:test_end]
    # start_time_train = timestamp[:test_start] + timestamp[test_end:]
    # d_test = duration[test_start:test_end]
    #
    # test_start = int(start_time_test[0] * sample_rate)
    # test_end = int((start_time_test[-1] + d_test[-1]) * sample_rate)
    #
    # data_test = data[:, test_start:test_end]
    # data_train = torch.cat((data[:, :test_start], data[:, test_end:]), dim=1)
    # time_test = times[test_start:test_end]
    # time_train = torch.cat((times[:test_start], times[test_end:]), dim=0)
    #
    # # eeg to event to frame
    # fps = args.fps
    # time_test = (time_test * fps).to(torch.int32)
    # time_train = (time_train * fps).to(torch.int32)
    # time_train_offset = time_train.clone()
    # time_train_offset[test_start:] -= ((times[test_end] - times[test_start]) * fps).to(torch.int32)  # bridge the gap to become continuous
    #
    # frames_test = eeg2frame(data_test, time_test, args.C)
    # frames_train = eeg2frame(data_train, time_train_offset, args.C)
    #
    # # frame corresponds to timestamp
    # time_test = torch.arange(time_test.min(), time_test.max() + 1) / fps  # index of time corresponds to index of frames
    # assert len(time_test) == frames_test.shape[0]
    # time_train = torch.arange(time_train_offset.min(), time_train_offset.max() + 1)
    # offset_index = (time_train == time_train_offset[test_start]).nonzero(as_tuple=True)[0].item()
    # time_train[offset_index:] += ((times[test_end] - times[test_start]) * fps).to(torch.int32)
    # time_train = time_train.to(float) / fps
    # assert len(time_train) == frames_train.shape[0]
    #
    # # block start time timestamp, expected spike timestamp
    # start_time_test = (torch.tensor(start_time_test) * fps).to(torch.int32).to(float) / fps
    # start_time_train = (torch.tensor(start_time_train) * fps).to(torch.int32).to(float) / fps

    ##################################################################################

    frames = {'test': frames_test, 'train': frames_train}
    times = {'test': time_test, 'train': time_train}
    spike_times = {'test': start_time_test, 'train': start_time_train}
    eegs = {'test': data_test, 'train': data_train}
    return frames, times, spike_times, eegs


def event_save(base_path, args):
    files = os.listdir(base_path)
    subject_idxes = [file for file in files if "run" in file]
    subject_idxes = np.array([subject_idx.split('_')[0] for subject_idx in subject_idxes])
    subject_idxes = np.unique(subject_idxes).tolist()

    frames_train_list, frames_test_list = [], []
    times_train_list, times_test_list = [], []
    spike_times_train_list, spike_times_test_list = [], []
    eeg_train_list, eeg_test_list = [], []
    subjects_train_list, subjects_test_list = [], []
    for i, idx in enumerate(subject_idxes):
        print(f"Processing {i}-th subject {idx}")
        for run in range(1, 21):
            frames, times, spike_times, eegs = process_subject_event(base_path, idx, run, args)
            frames_train_list.append(frames['train'])
            frames_test_list.append(frames['test'])
            times_train_list.append(times['train'])
            times_test_list.append(times['test'])
            spike_times_train_list.append(spike_times['train'])
            spike_times_test_list.append(spike_times['test'])
            eeg_train_list.append(eegs['train'])
            eeg_test_list.append(eegs['test'])
            subjects_train_list.append((i, (run - 1)))
            subjects_test_list.append((i, (run - 1)))

    N = len(frames_train_list)
    perm = np.random.RandomState(1).permutation(N)
    frames_train_list = [frames_train_list[i] for i in perm]
    times_train_list = [times_train_list[i] for i in perm]
    spike_times_train_list = [spike_times_train_list[i] for i in perm]
    eeg_train_list = [eeg_train_list[i] for i in perm]
    subjects_train_list = [subjects_train_list[i] for i in perm]

    N = len(frames_test_list)
    perm = np.random.RandomState(1).permutation(N)
    frames_test_list = [frames_test_list[i] for i in perm]
    times_test_list = [times_test_list[i] for i in perm]
    spike_times_test_list = [spike_times_test_list[i] for i in perm]
    eeg_test_list = [eeg_test_list[i] for i in perm]
    subjects_test_list = [subjects_test_list[i] for i in perm]

    train_dataset = {'frames': frames_train_list, 'times': times_train_list, 'spike_times': spike_times_train_list, 'eeg': eeg_train_list, 'subjects': subjects_train_list}
    test_dataset = {'frames': frames_test_list, 'times': times_test_list, 'spike_times': spike_times_test_list, 'eeg': eeg_test_list, 'subjects': subjects_test_list}

    dataset = broderick2019_event(train_dataset, base_path, args)
    save_path = os.path.join(base_path, rf"frames_n{args.n_frames}_s{args.stride}\train.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(dataset, f=save_path)
    print("Training data saved")
    del dataset

    dataset = broderick2019_event(test_dataset, base_path, args)
    save_path = os.path.join(base_path, rf"frames_n{args.n_frames}_s{args.stride}\test.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(dataset, f=save_path)
    print("Test data saved")
    del dataset


def generate_event_dataloader(base_path, batch_size, n_frames, stride, evaluate=False):
    if evaluate:
        dataloaders = {'test': None}
    else:
        dataloaders = {'train': None, 'test': None}
    for split, _ in dataloaders.items():
        dataset_path = os.path.join(base_path, f"frames_n{n_frames}_s{stride}", f"{split}.pt")
        dataset = torch.load(dataset_path, map_location='cpu', weights_only=False)
        dataloaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    return dataloaders


class broderick2019_event(Dataset):
    def __init__(self, data, base_path, args):
        n_frames, stride, fps = args.n_frames, args.stride, args.fps
        self.base_path = base_path if base_path else r"E:\NIPS2026\Dataset\broderick2019"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model_name = "facebook/wav2vec2-base-10k-voxpopuli"
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.w2v_model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        self.w2v_model.eval()

        self.frames = data['frames']
        self.times = data['times']
        self.spike_times = data['spike_times']
        self.eegs = data['eeg']
        self.subjects = data['subjects']

        # self.num_subjects = len(self.frames) // 20
        self.subjects_n, self.frames_n, self.times_n, self.spike_idxes_n, self.speech_reps_n, self.eeg_n = [], [], [], [], [], []
        for i in tqdm(range(len(self.subjects))):
            frames_i = self.frames[i]
            times_i = self.times[i]
            spike_times_i = self.spike_times[i]
            eeg_i = self.eegs[i]
            subjects_i = self.subjects[i]

            start_idx, stop_idx = 0, n_frames
            while stop_idx <= frames_i.shape[0]:
                selected_times_i = times_i[start_idx:stop_idx]
                spike_idxes = torch.tensor([(selected_times_i == t).nonzero()[0].item() if t in selected_times_i else -1 for t in spike_times_i])

                if (spike_idxes != -1).nonzero().numel() == 0 or spike_idxes[(spike_idxes != -1).nonzero()][0].min() <= 10 or spike_idxes.max() == n_frames:
                    start_idx += stride
                    stop_idx += stride
                    continue

                start_time, end_time = selected_times_i[0].item(), selected_times_i[-1].item()
                _, rep_audio = self.wav2vec(subjects_i[0], subjects_i[1], start_time, start_time + (n_frames - 1) / fps)

                eeg_start = int(start_idx / fps) * 120
                eeg_end = int(stop_idx / fps) * 120
                selected_eeg_i = eeg_i[:, eeg_start:eeg_end]

                if rep_audio is None or start_time == 0 or (end_time - start_time) > (n_frames / fps) or selected_eeg_i.shape[-1] != (eeg_end - eeg_start):
                    start_idx += stride
                    stop_idx += stride
                    continue
                rep_audio = rep_audio.permute(0, 2, 1).squeeze(0).cpu()  # [768, T1]
                # rep_audio = self.resample(rep_audio, n_frames // 2)

                self.subjects_n.append(subjects_i)
                self.frames_n.append(frames_i[start_idx:stop_idx, ...])
                self.times_n.append(selected_times_i)
                self.spike_idxes_n.append(spike_idxes)
                self.speech_reps_n.append(rep_audio)
                self.eeg_n.append(selected_eeg_i)

                start_idx += stride
                stop_idx += stride

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["w2v_model"]
        del state["feature_extractor"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        model_name = "facebook/wav2vec2-base-10k-voxpopuli"
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.w2v_model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        self.w2v_model.eval()

    def __len__(self):
        return len(self.subjects_n)

    def __getitem__(self, idx):
        return (
            self.subjects_n[idx],
            self.frames_n[idx],
            self.times_n[idx],
            torch.cat([self.spike_idxes_n[idx], torch.full((100 - len(self.spike_idxes_n[idx]),), -1)]),
            self.speech_reps_n[idx],
            self.eeg_n[idx],
        )

    def wav2vec(self, subject_id, run, start: float, stop: float):
        files = os.listdir(self.base_path)
        subject_idxes = [file for file in files if "run" in file]
        subject_idxes = np.array([subject_idx.split('_')[0] for subject_idx in subject_idxes])
        subject_idxes = np.unique(subject_idxes).tolist()
        subject_idx = subject_idxes[subject_id]

        event_path = os.path.join(self.base_path, f"{subject_idx}_run{run + 1}", 'events.csv')
        events = pd.read_csv(event_path)
        sound_events = events[events['kind'] == 'sound']
        sound_events.reset_index(drop=True, inplace=True)

        sound_start = np.array(sound_events['start'].tolist())
        index = (sound_start > start).argmax()
        index -= 1
        filepath = (sound_events.iloc[index])['filepath']
        start -= sound_start[index]
        stop -= sound_start[index]

        try:
            wav, sr = self.extract_wav(filepath, start, stop)
        except AssertionError:
            return None, None
        wav = torch.mean(wav, dim=0)

        model_sr = self.feature_extractor.sampling_rate
        wav = julius.resample.ResampleFrac(old_sr=int(sr), new_sr=model_sr)(wav)

        # [1, T]
        processed_wav = self.feature_extractor(wav, return_tensors="pt", sampling_rate=model_sr, do_normalize=True).input_values
        with torch.no_grad():
            outputs = self.w2v_model(processed_wav.to(self.device), output_hidden_states=True)
        hidden_states = outputs.get("hidden_states")
        last_hidden_state = outputs.get("last_hidden_state")
        if isinstance(hidden_states, tuple):
            hidden_states = torch.stack(hidden_states)
        # hidden_states[0] is equal to last_hidden_state
        return hidden_states, last_hidden_state

    def extract_wav(self, filepath, onset: float, offset: float):
        try:
            info = torchaudio.info(str(filepath))
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
            rep = resample_poly(rep, up=sample_num, down=rep.shape[-1], axis=-1)
            return torch.tensor(rep)


def eeg_save(base_path):
    files = os.listdir(base_path)
    subject_idxes = [file for file in files if "run" in file]
    subject_idxes = np.array([subject_idx.split('_')[0] for subject_idx in subject_idxes])
    subject_idxes = np.unique(subject_idxes).tolist()

    eeg_list = []
    for i, idx in enumerate(subject_idxes):
        print(f"Processing {i}-th subject {idx}")
        for run in range(1, 21):
            files = os.listdir(os.path.join(base_path, f"{idx}_run{run}"))
            for file in files:
                if file.endswith('.fif'):
                    fif_file = file
            fif_path = os.path.join(base_path, f"{idx}_run{run}", fif_file)

            sample_rate = re.findall(r'sr(\d+)-', fif_file)
            sample_rate = int(sample_rate[0]) if sample_rate else 0
            raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=0)
            data, times = raw[:, :]
            data = torch.tensor(data).float().cpu()
            eeg_list.append(data)

    dataset = broderick2019_eeg(eeg_list, sample_rate)
    save_path = os.path.join(base_path, rf"eeg.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(dataset, f=save_path)
    print("eeg data saved")
    del dataset


class broderick2019_eeg(Dataset):
    def __init__(self, data, sample_rate):
        self.data = data
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        subject_idx, run = idx[0], idx[1]
        return self.data[subject_idx * 20 + run], self.sample_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--C', type=float, default=0.2)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--p', type=float, default=0.3)
    parser.add_argument('--n_frames', type=int, default=50)
    parser.add_argument('--stride', type=int, default=10)
    args = parser.parse_args()

    path = r"/datasets/broderick2019"
    event_save(path, args)
    eeg_save(path)

    dataloaders = generate_event_dataloader(path, 16, args.n_frames, args.stride, evaluate=False)
    train_num, test_num = 0, 0
    for data_batch in dataloaders['train']:
        subject_ids, frames, times, expect_spike_idxes, speech_reps, eegs = data_batch
        train_num += frames.shape[0]
    for data_batch in dataloaders['test']:
        subject_ids, frames, times, expect_spike_idxes, speech_reps, eegs = data_batch
        test_num += frames.shape[0]
    print(f"Sample numbers {train_num}/{test_num} (train/test)")


