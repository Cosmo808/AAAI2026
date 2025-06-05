import logging
import os
import pdb
import argparse
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
import random

logger = logging.getLogger(__name__)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Model

# model_name = "facebook/wav2vec2-large-xlsr-53"
# model_name = "facebook/wav2vec2-base-960h"
model_name = "facebook/wav2vec2-base-10k-voxpopuli"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name)
model.to(device)
model.eval()


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
    # print(
    #     "Preprocessing Wav on %s, start %.1f, stop %.1f, duration %.1f",
    #     filepath, start, stop, stop - start)
    wav = torch.mean(wav, dim=0)  # stereo to mono

    model_sr = feature_extractor.sampling_rate
    wav = julius.resample.ResampleFrac(old_sr=int(sr), new_sr=model_sr)(wav)

    # [1, T]
    processed_wav = feature_extractor(wav, return_tensors="pt", sampling_rate=model_sr, do_normalize=True).input_values
    with torch.no_grad():
        outputs = model(processed_wav.to(device), output_hidden_states=True)
    hidden_states = outputs.get("hidden_states")
    last_hidden_state = outputs.get("last_hidden_state")
    if isinstance(hidden_states, tuple):
        hidden_states = torch.stack(hidden_states)
    # hidden_states[0] is equal to last_hidden_state
    return hidden_states, last_hidden_state


def process_subject(base_path, idx, length=None, sample_t=None):
    start_stop, speech_seg, speech_rep, eeg_seg = [], [], [], []

    for run in range(1, 21):
        # obtain file path
        try:
            files = os.listdir(os.path.join(base_path, f"{idx}_run{run}"))
        except FileNotFoundError:
            continue
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
        data, times = raw[:, :]  # data size [128, 21514]
        events = pd.read_csv(event_path)
        sound_events = events[events['kind'] == 'sound']
        block_events = events[events['kind'] == 'block']
        word_events = events[events['kind'] == 'word']
        sound_events.reset_index(drop=True, inplace=True)
        block_events.reset_index(drop=True, inplace=True)
        word_events.reset_index(drop=True, inplace=True)

        # segmentation
        if length is not None:
            timestamp = np.array(word_events['start'].tolist())
            duration = np.array(word_events['duration'].tolist())
            word = word_events['word'].tolist()

            for i in range(0, len(timestamp), 20):
                start_timestamp = timestamp[i]
                stop_timestamp = start_timestamp + length
                if stop_timestamp > timestamp[-1] + duration[-1]:
                    break

                _, rep = wav2vec(sound_event=sound_events, start=start_timestamp, stop=stop_timestamp)
                if rep is None:
                    continue
                s = int(start_timestamp * sample_rate)
                sample_num = int(sample_rate * length)
                eeg = torch.tensor(data[:, s:(s + sample_num)]).float().cpu()
                if eeg.shape[-1] == 0:
                    continue
                eeg_seg.append(eeg)
                rep = rep.permute(0, 2, 1)
                rep = F.interpolate(rep, size=sample_num)
                speech_rep.append(rep.squeeze(0).detach().cpu())
                del eeg, rep

                start_stop.append([start_timestamp, stop_timestamp])
                start_index = np.where(timestamp == start_timestamp)[0][0]
                index = (timestamp > stop_timestamp).argmax()
                speech_seg.append(' '.join(word[start_index:index]))

        else:
            resample_length = int(sample_t * sample_rate)
            timestamp = np.array(block_events['start'].tolist())
            duration = np.array(block_events['duration'].tolist())
            speech = block_events['uid'].tolist()
            sample_num = 0

            for i, (s, d, s_seg) in enumerate(zip(timestamp, duration, speech)):
                if d == np.inf:
                    d = word_events['start'].tolist()[-1] + word_events['duration'].tolist()[-1] - s

                # if d >= 2 * sample_t:
                if d >= 10:
                    continue

                _, rep = wav2vec(sound_event=sound_events, start=s, stop=(s + d))
                if rep is not None:
                    sample_num += 1
                    rep = rep.permute(0, 2, 1)
                    rep = F.interpolate(rep, size=resample_length)
                    speech_rep.append(rep.squeeze(0).detach().cpu())
                    start_stop.append([s, s + d])
                    speech_seg.append(s_seg)
                    slice_data = torch.tensor(data[:, int(s * sample_rate):int((s + d) * sample_rate)])
                    resample_data = resample(slice_data, sample_num=resample_length)
                    eeg_seg.append(resample_data.cpu())
            # print(f"run{run} has {sample_num} samples")

    return start_stop, speech_seg, speech_rep, eeg_seg


def save_datasets(base_path, split_method=None, length=None, sample_t=None):
    if split_method is None:
        split_method = ('half', 0)
    files = os.listdir(base_path)
    subject_idxes = [file for file in files if "run" in file]
    subject_idxes = np.array([subject_idx.split('_')[0] for subject_idx in subject_idxes])
    subject_idxes = np.unique(subject_idxes).tolist()

    subject_id_list, start_stop_list, speech_seg_list, speech_rep_list, eeg_seg_list = [], [], [], [], []
    for i, idx in enumerate(subject_idxes):
        start_stop, speech_seg, speech_rep, eeg_seg = process_subject(base_path, idx, length, sample_t)
        print(f"Processed {i}-th subject {idx} with {len(start_stop)} samples")
        subject_id_list.extend([subject_idxes.index(idx) for i in range(len(start_stop))])
        start_stop_list.extend(start_stop)
        speech_seg_list.extend(speech_seg)
        speech_rep_list.extend(speech_rep)
        eeg_seg_list.extend(eeg_seg)
        # if i == 1:
        #     break

###########################################################################
    # for adaptive slicing
    if length is None:
        run_sizes = [41, 55, 48, 54, 56,
                     48, 41, 16, 27, 36,
                     30, 31, 46, 43, 43,
                     26, 42, 46, 32, 58]
        n_subjects = 19
        samples_per_subject = 819
        n_folds = 5
        fold = split_method[1]

        train_idx = []
        test_idx = []
        for subj in range(n_subjects):
            subj_offset = subj * samples_per_subject
            for run, run_size in enumerate(run_sizes):
                offset = subj_offset + sum(run_sizes[:run])
                indices = list(range(offset, offset + run_size))

                fold_size = run_size // n_folds
                random.seed(int(subj) + int(run))

                test_inds = random.sample(indices, fold_size)
                train_inds = list(set(indices) - set(test_inds))

                test_idx.append(test_inds)
                train_idx.append(train_inds)

        random.shuffle(train_idx)
        train_idx = [idx for sublist in train_idx for idx in sublist]
        random.shuffle(test_idx)
        test_idx = [idx for sublist in test_idx for idx in sublist]

        train_subject_id = [subject_id_list[i] for i in train_idx]
        train_start_stop = [start_stop_list[i] for i in train_idx]
        train_speech_seg = [speech_seg_list[i] for i in train_idx]
        train_speech_rep = [speech_rep_list[i] for i in train_idx]
        train_eeg_seg = [eeg_seg_list[i] for i in train_idx]
        train_dataset = broderick2019(train_subject_id, train_start_stop, train_speech_seg, train_speech_rep, train_eeg_seg)

        test_subject_id = [subject_id_list[i] for i in test_idx]
        test_start_stop = [start_stop_list[i] for i in test_idx]
        test_speech_seg = [speech_seg_list[i] for i in test_idx]
        test_speech_rep = [speech_rep_list[i] for i in test_idx]
        test_eeg_seg = [eeg_seg_list[i] for i in test_idx]
        test_dataset = broderick2019(test_subject_id, test_start_stop, test_speech_seg, test_speech_rep, test_eeg_seg)

###########################################################################
    # shuffle split
    else:
        shuffled_idx = np.random.permutation(len(subject_id_list))
        subject_id_list = [subject_id_list[i] for i in shuffled_idx]
        start_stop_list = [start_stop_list[i] for i in shuffled_idx]
        speech_seg_list = [speech_seg_list[i] for i in shuffled_idx]
        speech_rep_list = [speech_rep_list[i] for i in shuffled_idx]
        eeg_seg_list = [eeg_seg_list[i] for i in shuffled_idx]

        if split_method[0] == 'half':
            num = len(subject_id_list) // 2
            train = broderick2019(subject_id_list[:num], start_stop_list[:num], speech_seg_list[:num], speech_rep_list[:num], eeg_seg_list[:num])
            test = broderick2019(subject_id_list[num:], start_stop_list[num:], speech_seg_list[num:], speech_rep_list[num:], eeg_seg_list[num:])
            if split_method[1] == 0:
                datasets = {'train': train, 'test': test}
            else:
                datasets = {'train': test, 'test': train}
            for split, dataset in datasets.items():
                save_path = os.path.join(base_path, fr"data_splits_l{length}\{split_method[0]}{split_method[1]}_{split}.pt")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(dataset, f=save_path)
        elif split_method[0] == '5fold':
            fold_size = len(subject_id_list) // 5
            fold = split_method[1]
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < 4 else len(subject_id_list)
            test_subject_id = subject_id_list[test_start:test_end]
            test_start_stop = start_stop_list[test_start:test_end]
            test_speech_seg = speech_seg_list[test_start:test_end]
            test_speech_rep = speech_rep_list[test_start:test_end]
            test_eeg_seg = eeg_seg_list[test_start:test_end]
            test_dataset = broderick2019(test_subject_id, test_start_stop, test_speech_seg, test_speech_rep, test_eeg_seg)

            train_subject_id = subject_id_list[:test_start] + subject_id_list[test_end:]
            train_start_stop = start_stop_list[:test_start] + start_stop_list[test_end:]
            train_speech_seg = speech_seg_list[:test_start] + speech_seg_list[test_end:]
            train_speech_rep = speech_rep_list[:test_start] + speech_rep_list[test_end:]
            train_eeg_seg = eeg_seg_list[:test_start] + eeg_seg_list[test_end:]
            train_dataset = broderick2019(train_subject_id, train_start_stop, train_speech_seg, train_speech_rep, train_eeg_seg)

###########################################################################

    datasets = {'train': train_dataset, 'test': test_dataset}
    for split, dataset in datasets.items():
        save_path = os.path.join(base_path, fr"data_splits_l{length}\{split_method[0]}{fold}_{split}.pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(dataset, f=save_path)


def generate_dataloader(base_path, batch_size, split_method, length, evaluate=False):
    split_method = split_method[0] + str(split_method[1])
    if evaluate:
        dataloaders = {'test': None}
    else:
        dataloaders = {'train': None, 'test': None}
    for split, _ in dataloaders.items():
        dataset_path = os.path.join(base_path, f"data_splits_l{length}", f"{split_method}_{split}.pt")
        dataset = torch.load(dataset_path, weights_only=False)
        dataloaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return dataloaders


class broderick2019(Dataset):
    def __init__(self, subject_id_list, start_stop_list, speech_seg_list, speech_rep_list, eeg_seg_list):
        self.subject_id_list = subject_id_list
        self.start_stop_list = start_stop_list
        self.speech_seg_list = speech_seg_list
        self.speech_rep_list = speech_rep_list
        self.eeg_seg_list = eeg_seg_list

        self.electrode_num = self.eeg_seg_list[0].shape[0]
        self.feature_dim = self.speech_rep_list[0].shape[1]

    def __len__(self):
        return len(self.start_stop_list)

    def __getitem__(self, idx):
        # Return a tuple of all four lists (you can modify this depending on your use case)
        return (self.subject_id_list[idx],
                self.start_stop_list[idx],
                self.speech_rep_list[idx],
                self.eeg_seg_list[idx])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--length', type=float, default=None)
    args = parser.parse_args()

    path = r"../../datasets/broderick2019"
    split_method = ('5fold', 0)
    # args.length = 5

    save_datasets(base_path=path, split_method=split_method, length=args.length, sample_t=5)

    dataloaders = generate_dataloader(base_path=path, batch_size=16, split_method=split_method, length=args.length)
    train_num, test_num = 0, 0
    for data_batch in dataloaders['train']:
        _, _, speech_rep, eeg_seg = data_batch
        in_channels = eeg_seg[0].shape[0]
        feature_dim = speech_rep[0].shape[0]
        train_num += eeg_seg.shape[0]
    for data_batch in dataloaders['test']:
        _, _, _, eeg_seg = data_batch
        test_num += eeg_seg.shape[0]
    print(f"Sample numbers {train_num}/{test_num} (train/test)")
