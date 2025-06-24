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
    'sub-A2002', 'sub-A2003', 'sub-A2004', 'sub-A2005', 'sub-A2006',
    'sub-A2007', 'sub-A2008', 'sub-A2009', 'sub-A2010', 'sub-A2013',
    'sub-A2014', 'sub-A2015', 'sub-A2016', 'sub-A2017', 'sub-A2019',
    'sub-A2020', 'sub-A2021', 'sub-A2024', 'sub-A2025', 'sub-A2027',
    'sub-A2028', 'sub-A2029', 'sub-A2030', 'sub-A2031', 'sub-A2032',
    'sub-A2033', 'sub-A2034', 'sub-A2035', 'sub-A2037', 'sub-A2038'
]
sample_rate = 120
sample_t = 5
seq_length = 1

wav = wav_processor(model_name='facebook/wav2vec2-large-xlsr-53')


class Param:
    pass


param = Param()
param.C = 0.2
param.fps = 4
param.sr = sample_rate
b2e = Brain2Event(param)

base_dir = r"E:\NIPS2026\datasets\schoffelen2019"

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
        # block_events = events[events['kind'] == 'block']
        word_events = events[events['kind'] == 'word']
        sound_events.reset_index(drop=True, inplace=True)
        sound_events.reset_index(drop=True, inplace=True)
        word_events.reset_index(drop=True, inplace=True)

        resample_length = int(sample_t * sample_rate)
        timestamp = np.array(sound_events['start'].tolist())
        duration = np.array(sound_events['duration'].tolist())
        speech = sound_events['word_sequence'].tolist()
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
                eegs = [eegs[i] for i in range(1, seq_length)]
                labels = [labels[i] for i in range(1, seq_length)]
                texts = [texts[i] for i in range(1, seq_length)]
                num += 1


if __name__ == '__main__':
    main()
