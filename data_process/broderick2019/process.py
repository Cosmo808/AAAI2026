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


subjects_id = list(range(1, 20))
runs_id = list(range(1, 21))
sample_rate = 120
sample_t = 5
seq_length = 5

wav = wav_processor()

class Param:
    pass
param = Param()
param.C = 0.2
param.fps = 5
param.sr = sample_rate
b2e = Brain2Event(param)

base_dir = r"E:\NIPS2026\datasets\broderick2019"
seq_dir = rf'{base_dir}\seq'
label_dir = rf'{base_dir}\labels'
event_dir = rf'{base_dir}\events'


def main():
    for subject in subjects_id:
        print(f"Processing subject {subject}")
        for run in runs_id:
            print(f"+++ run {run}")
            eeg_file = os.path.join(base_dir, fr"{subject}_run{run}\meg-sr120-hp0-raw.fif")
            event_file = os.path.join(base_dir, fr"{subject}_run{run}\events.csv")
            os.makedirs(rf"{seq_dir}\s{subject}_r{run}", exist_ok=True)
            os.makedirs(rf"{label_dir}\s{subject}_r{run}", exist_ok=True)
            os.makedirs(rf"{event_dir}\s{subject}_r{run}", exist_ok=True)

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
            num = 0

            eegs = []
            labels = []
            for i, (s, d) in enumerate(zip(timestamp, duration)):
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

                if len(eegs) == seq_length:
                    eegs = torch.stack(eegs)
                    labels = torch.stack(labels)
                    events = b2e.forward(eegs)
                    # torch.save(eegs, rf"{seq_dir}\s{subject}_r{run}\{num}.pth")
                    # torch.save(labels, rf"{label_dir}\s{subject}_r{run}\{num}.pth")
                    torch.save(events, rf"{event_dir}\s{subject}_r{run}\{num}.pth")
                    eegs, labels = [], []
                    num += 1


if __name__ == '__main__':
    main()