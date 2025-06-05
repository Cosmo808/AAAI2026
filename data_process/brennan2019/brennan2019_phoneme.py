import logging
import os
import pdb
import re
import mne
import pandas as pd
import typing as tp
from pathlib import Path
from typing import Union
import julius
import torch
import torchaudio
import torch.nn.functional as F
from scipy.interpolate import splrep, splev
from scipy.signal import resample_poly
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Model
# model_name = "facebook/wav2vec2-large-xlsr-53"
# model_name = "facebook/wav2vec2-base-960h"
model_name = "facebook/wav2vec2-base-10k-voxpopuli"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
wav2vec_model = Wav2Vec2Model.from_pretrained(model_name)
wav2vec_model.to(device)
wav2vec_model.eval()

from transformers import AutoModel, AutoProcessor, AutoModelForPreTraining
upstream_model_card = "facebook/wav2vec2-large-lv60"
reborn_model_card = "andybi7676/reborn-uasr_ls100h_iter5-stage1"
processor = AutoProcessor.from_pretrained(upstream_model_card)
upstream_model = AutoModelForPreTraining.from_pretrained(upstream_model_card)
reborn_model = AutoModel.from_pretrained(reborn_model_card, trust_remote_code=True, revision="main")
upstream_model = upstream_model.to(device)
reborn_model = reborn_model.to(device)
upstream_model.eval()
reborn_model.eval()


def resample(rep, sample_num):
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


def extract_wav(filepath: Union[Path, str], onset: float, offset: float) -> tp.Tuple[torch.Tensor, float]:
    info = torchaudio.info(str(filepath))
    sr = float(info.sample_rate)
    frame_offset = np.round(onset * sr).astype(int) if isinstance(onset, np.ndarray) else int(round(onset * sr))
    num_frames = np.round((offset - onset) * sr).astype(int) if isinstance((offset - onset), np.ndarray) else int(
        round((offset - onset) * sr))
    wav = torchaudio.load(filepath, frame_offset=frame_offset, num_frames=num_frames)[0]
    delta = abs(wav.shape[-1] / sr - offset + onset)
    assert delta <= 0.1, (delta, filepath, onset, offset, onset - offset)
    return wav, sr


def wav2vecU_segment(sound_event, start: float, stop: float, min_t):
    if (stop - start) < min_t:
        return [start, stop]

    sound_start = np.array(sound_event['start'].tolist())
    index = (sound_start > start).argmax()
    index -= 1
    filepath = (sound_event.iloc[index])['filepath']
    start -= sound_start[index]
    stop -= sound_start[index]

    try:
        wav, sr = extract_wav(filepath, start, stop)
    except AssertionError:
        return [start + sound_start[index], stop + sound_start[index]]

    wav = torch.mean(wav, dim=0)
    model_sr = 16000
    wav = julius.resample.ResampleFrac(old_sr=int(sr), new_sr=model_sr)(wav)

    processed_wav = processor(wav, return_tensors="pt", padding="longest", sampling_rate=model_sr).input_values
    with torch.no_grad():
        outputs = upstream_model(processed_wav.to(device), output_hidden_states=True)
    hidden_states = outputs.get("hidden_states")
    last_hidden_state = outputs.get("last_hidden_state")
    if last_hidden_state is None:
        last_hidden_state = hidden_states[15]

    padding_mask = torch.zeros(last_hidden_state.shape[:-1], dtype=torch.bool, device=device)
    boundary = reborn_model.forward(last_hidden_state, padding_mask)['boundary'][0]   # [T]

    sliced_times = [start]
    unit = (stop - start) / boundary.shape[-1]
    i, stride = round(min_t / unit), round(min_t / unit)
    while i < boundary.shape[-1]:
        if boundary[i] == 1:
            sliced_times.append(start + i * unit)
            i += stride
        else:
            i += 1
    if stop - sliced_times[-1] < min_t:
        sliced_times[-1] = stop
    else:
        sliced_times.append(stop)
    sliced_times = [t + sound_start[index] for t in sliced_times]
    return sliced_times


def wav2vec(sound_event, start: float, stop: float):
    sound_start = np.array(sound_event['start'].tolist())
    index = (sound_start > start).argmax()
    index -= 1
    filepath = (sound_event.iloc[index])['filepath']
    start -= sound_start[index]
    stop -= sound_start[index]

    try:
        wav, sr = extract_wav(filepath, start, stop)
    except AssertionError:
        return None, None

    wav = torch.mean(wav, dim=0)  # stereo to mono
    model_sr = feature_extractor.sampling_rate
    wav = julius.resample.ResampleFrac(old_sr=int(sr), new_sr=model_sr)(wav)

    # [1, T]
    processed_wav = feature_extractor(wav, return_tensors="pt", sampling_rate=model_sr, do_normalize=True).input_values
    with torch.no_grad():
        outputs = wav2vec_model(processed_wav.to(device), output_hidden_states=True)
    hidden_states = outputs.get("hidden_states")
    last_hidden_state = outputs.get("last_hidden_state")
    if isinstance(hidden_states, tuple):
        hidden_states = torch.stack(hidden_states)
    # hidden_states[0] is equal to last_hidden_state
    return hidden_states, last_hidden_state


def process_subject(base_path, idx, sample_t=None, min_t=0):
    # obtain file path
    files = os.listdir(os.path.join(base_path, str(idx)))
    for file in files:
        if file.endswith('.fif'):
            fif_file = file
        elif file.endswith('.csv'):
            event_file = file
    fif_path = os.path.join(base_path, str(idx), fif_file)
    event_path = os.path.join(base_path, str(idx), event_file)

    # load eeg and event data
    sample_rate = re.findall(r'sr(\d+)-', fif_file)
    sample_rate = int(sample_rate[0]) if sample_rate else 0
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=0)
    data, times = raw[:, :]
    events = pd.read_csv(event_path)
    sound_events = events[events['kind'] == 'sound']
    block_events = events[events['kind'] == 'block']
    word_events = events[events['kind'] == 'word']
    sound_events.reset_index(drop=True, inplace=True)
    block_events.reset_index(drop=True, inplace=True)
    word_events.reset_index(drop=True, inplace=True)

    # segmentation
    start_stop, speech_rep, eeg_seg = [], [], []
    resample_length = int(sample_t * sample_rate)
    timestamp = np.array(block_events['start'].tolist())
    duration = np.array(block_events['duration'].tolist())
    speech = block_events['uid'].tolist()

    ss, ds = [], []
    concat_s, concat_d = None, None
    for i, (s, d) in enumerate(zip(timestamp, duration)):
        if d == np.inf:
            d = word_events['start'].tolist()[-1] + word_events['duration'].tolist()[-1] - s

        if min_t <= d < 2 * sample_t:
            ss.append(s)
            ds.append(d)
        elif d < min_t:
            if concat_s is None:
                # ss.append(s)
                # ds.append(d)
                concat_s, concat_d = s, d
            while i + 1 < len(timestamp):
                i += 1
                s_next, d_next, s_seg_next = timestamp[i], duration[i], speech[i]
                concat_d += d_next
                if min_t <= concat_d < 2 * sample_t:
                    ss.append(s)
                    ds.append(concat_d)
                else:
                    break
            concat_s, concat_d = None, None

    for s, d in zip(ss, ds):
        sliced_times = wav2vecU_segment(sound_event=sound_events, start=s, stop=(s + d), min_t=min_t / 2)
        for i in range(len(sliced_times) - 1):
            for j in range(i + 1, len(sliced_times)):
                start = sliced_times[i]
                stop = sliced_times[j]
                _, rep = wav2vec(sound_events, start, stop)
                if rep is not None:
                    rep = rep.permute(0, 2, 1)
                    rep = F.interpolate(rep, size=resample_length)
                    speech_rep.append(rep.squeeze(0).detach().cpu())
                    start_stop.append([start, stop])
                    slice_data = torch.tensor(data[:, int(start * sample_rate):int(stop * sample_rate)])
                    resample_data = resample(slice_data, sample_num=resample_length)
                    eeg_seg.append(resample_data.cpu())

    return start_stop, speech_rep, eeg_seg


def save_datasets(base_path, split_method=None, sample_t=None, min_t=0):
    if split_method is None:
        split_method = ('half', 0)
    files = os.listdir(base_path)
    subject_idxes = [file for file in files if "S" in file]

    subject_id_list, start_stop_list, speech_rep_list, eeg_seg_list = [], [], [], []
    for i, idx in enumerate(subject_idxes):
        start_stop, speech_rep, eeg_seg = process_subject(base_path, idx, sample_t, min_t)
        print(f"Processed {i}-th subject {idx} with {len(start_stop)} samples")
        subject_id_list.extend([subject_idxes.index(idx) for i in range(len(start_stop))])
        start_stop_list.extend(start_stop)
        speech_rep_list.extend(speech_rep)
        eeg_seg_list.extend(eeg_seg)

    shuffled_idx = np.random.permutation(len(subject_id_list))
    subject_id_list = [subject_id_list[i] for i in shuffled_idx]
    start_stop_list = [start_stop_list[i] for i in shuffled_idx]
    speech_rep_list = [speech_rep_list[i] for i in shuffled_idx]
    eeg_seg_list = [eeg_seg_list[i] for i in shuffled_idx]

    if split_method[0] == 'half':
        num = len(subject_id_list) // 2
        train = brennan2019(subject_id_list[:num], start_stop_list[:num], speech_rep_list[:num], eeg_seg_list[:num])
        test = brennan2019(subject_id_list[num:], start_stop_list[num:], speech_rep_list[num:], eeg_seg_list[num:])
        if split_method[1] == 0:
            datasets = {'train': train, 'test': test}
        else:
            datasets = {'train': test, 'test': train}
        for split, dataset in datasets.items():
            save_path = os.path.join(base_path, fr"data_splits_l{length}_phoneme\{split_method[0]}{split_method[1]}_{split}.pt")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(dataset, f=save_path)
    elif split_method[0] == '5fold':
        fold_size = len(subject_id_list) // 5
        fold = split_method[1]
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < 4 else len(subject_id_list)
        test_subject_id = subject_id_list[test_start:test_end]
        test_start_stop = start_stop_list[test_start:test_end]
        test_speech_rep = speech_rep_list[test_start:test_end]
        test_eeg_seg = eeg_seg_list[test_start:test_end]
        test_dataset = brennan2019(test_subject_id, test_start_stop, test_speech_rep, test_eeg_seg)

        train_subject_id = subject_id_list[:test_start] + subject_id_list[test_end:]
        train_start_stop = start_stop_list[:test_start] + start_stop_list[test_end:]
        train_speech_rep = speech_rep_list[:test_start] + speech_rep_list[test_end:]
        train_eeg_seg = eeg_seg_list[:test_start] + eeg_seg_list[test_end:]
        train_dataset = brennan2019(train_subject_id, train_start_stop, train_speech_rep, train_eeg_seg)

        datasets = {'train': train_dataset, 'test': test_dataset}
        for split, dataset in datasets.items():
            save_path = os.path.join(base_path, fr"data_splits_l{length}_phoneme\{split_method[0]}{fold}_{split}.pt")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(dataset, f=save_path)


def generate_dataloader(base_path, batch_size, split_method, length, evaluate=False):
    split_method = split_method[0] + str(split_method[1])
    if evaluate:
        dataloaders = {'test': None}
    else:
        dataloaders = {'train': None, 'test': None}
    for split, _ in dataloaders.items():
        dataset_path = os.path.join(base_path, f"data_splits_l{length}_phoneme", f"{split_method}_{split}.pt")
        dataset = torch.load(dataset_path, weights_only=False)
        dataloaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return dataloaders


class brennan2019(Dataset):
    def __init__(self, subject_id_list, start_stop_list, speech_rep_list, eeg_seg_list):
        self.subject_id_list = subject_id_list
        self.start_stop_list = start_stop_list
        self.speech_rep_list = speech_rep_list
        self.eeg_seg_list = eeg_seg_list

        self.electrode_num = self.eeg_seg_list[0].shape[0]
        self.feature_dim = self.speech_rep_list[0].shape[1]

    def __len__(self):
        return len(self.start_stop_list)

    def __getitem__(self, idx):
        return (self.subject_id_list[idx],
                self.start_stop_list[idx],
                self.speech_rep_list[idx],
                self.eeg_seg_list[idx])


if __name__ == "__main__":
    path = r"../../datasets/brennan2019"
    split_method = ('5fold', 0)
    length = None
    min_t = 5
    save_datasets(base_path=path, split_method=split_method, sample_t=8, min_t=min_t)

    dataloaders = generate_dataloader(base_path=path, batch_size=16, split_method=split_method, length=length)
    train_num, test_num = 0, 0
    for data_batch in dataloaders['train']:
        _, _, speech_rep, eeg_seg = data_batch
        in_channels = eeg_seg[0].shape[0]
        feature_dim = speech_rep[0].shape[0]
        train_num += eeg_seg.shape[0]
    for data_batch in dataloaders['test']:
        _, _, speech_rep, eeg_seg = data_batch
        test_num += eeg_seg.shape[0]
    print(f"Sample numbers {train_num}/{test_num} (train/test)")
