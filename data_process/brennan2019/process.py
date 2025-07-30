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
import torch.nn.functional as F
from models.utils import Brain2Event, wav_processor


subjects_id = [
    'S01', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10',
    'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19',
    'S20', 'S21', 'S22', 'S23', 'S25', 'S38', 'S39', 'S40', 'S41',
    'S42', 'S43', 'S44', 'S45', 'S48',
]
sample_rate = 120
sample_t = 8.334
seq_length = 5

wav = wav_processor()

class Param:
    pass
param = Param()
param.C = 0.2
param.fps = 3
param.sr = sample_rate
b2e = Brain2Event(param)

base_dir = r"yourpath\datasets\brennan2019"
seq_dir = rf'{base_dir}\seq'
label_dir = rf'{base_dir}\labels'
event_dir = rf'{base_dir}\events'
text_dir = rf'{base_dir}\texts'


def main():
    for subject in subjects_id:
        print(f"Processing subject {subject}")

        eeg_file = os.path.join(base_dir, fr"{subject}\meg-sr120-hp0-raw.fif")
        event_file = os.path.join(base_dir, fr"{subject}\events.csv")
        os.makedirs(rf"{seq_dir}\{subject}", exist_ok=True)
        os.makedirs(rf"{label_dir}\{subject}", exist_ok=True)
        os.makedirs(rf"{event_dir}\{subject}", exist_ok=True)
        os.makedirs(rf"{text_dir}\{subject}", exist_ok=True)

        raw = mne.io.read_raw_fif(eeg_file, preload=True, verbose=0)
        data, times = raw[:, :]

        events = pd.read_csv(event_file)
        sound_events = events[events['kind'] == 'sound']
        block_events = events[events['kind'] == 'block']
        word_events = events[events['kind'] == 'word']
        sound_events.reset_index(drop=True, inplace=True)
        block_events.reset_index(drop=True, inplace=True)
        word_events.reset_index(drop=True, inplace=True)

        resample_length = int(sample_t * sample_rate)
        timestamp = np.array(block_events['start'].tolist())
        duration = np.array(block_events['duration'].tolist())
        speech = block_events['uid'].tolist()
        num = 0

        eegs = []
        labels = []
        texts = []
        for i, (s, d, t) in enumerate(zip(timestamp, duration, speech)):
            if d == np.inf:
                d = word_events['start'].tolist()[-1] + word_events['duration'].tolist()[-1] - s

            if d >= 2 * sample_t:
                continue

            _, rep = wav.wav2vec(sound_event=sound_events, start=s, stop=(s + d))
            if rep is not None:
                rep = rep.permute(0, 2, 1)
                rep = F.interpolate(rep, size=resample_length)
                labels.append(rep.squeeze(0).detach().cpu())
                slice_data = torch.tensor(data[:, int(s * sample_rate):int((s + d) * sample_rate)])
                resample_data = wav.resample(slice_data, sample_num=resample_length)
                eegs.append(resample_data.cpu())
                texts.append(t)

            if len(texts) == seq_length:
                eegs = torch.stack(eegs)
                labels = torch.stack(labels)
                events = b2e.forward(eegs)
                torch.save(eegs, rf"{seq_dir}\{subject}\{num}.pth")
                torch.save(labels, rf"{label_dir}\{subject}\{num}.pth")
                torch.save(events, rf"{event_dir}\{subject}\{num}.pth")
                torch.save(texts, rf"{text_dir}\{subject}\{num}.pth")
                # eegs, labels = [], []
                eegs = [eegs[i] for i in range(1, seq_length)]
                labels = [labels[i] for i in range(1, seq_length)]
                texts = [texts[i] for i in range(1, seq_length)]
                num += 1

        # concat
        # random.seed(42)  # for reproducibility
        # combo_count = 0
        # used_combos = set()
        #
        # while combo_count < 90:
        #     indices = tuple(sorted(random.sample(range(len(timestamp)), seq_length)))
        #     if indices in used_combos:
        #         continue
        #     used_combos.add(indices)
        #
        #     eegs = []
        #     labels = []
        #
        #     for idx in indices:
        #         s = timestamp[idx]
        #         d = duration[idx]
        #         if d == np.inf:
        #             d = word_events['start'].tolist()[-1] + word_events['duration'].tolist()[-1] - s
        #         if d >= 2 * sample_t:
        #             continue
        #
        #         _, rep = wav.wav2vec(sound_event=sound_events, start=s, stop=(s + d))
        #         if rep is None:
        #             continue
        #
        #         rep = rep.permute(0, 2, 1)
        #         rep = F.interpolate(rep, size=resample_length)
        #         labels.append(rep.squeeze(0).detach().cpu())
        #
        #         slice_data = torch.tensor(data[:, int(s * sample_rate):int((s + d) * sample_rate)])
        #         resample_data = wav.resample(slice_data, sample_num=resample_length)
        #         eegs.append(resample_data.cpu())
        #
        #     if len(eegs) != seq_length:
        #         continue  # skip if any rep was None or d too long
        #
        #     eegs_tensor = torch.stack(eegs)
        #     labels_tensor = torch.stack(labels)
        #     events = b2e.forward(eegs_tensor)
        #
        #     torch.save(eegs_tensor, rf"{seq_dir}\{subject}\rand_{combo_count}.pth")
        #     torch.save(labels_tensor, rf"{label_dir}\{subject}\rand_{combo_count}.pth")
        #     torch.save(events, rf"{event_dir}\{subject}\rand_{combo_count}.pth")
        #
        #     combo_count += 1


if __name__ == '__main__':
    main()