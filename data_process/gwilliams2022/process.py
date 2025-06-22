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


subjects_id = [f"{i:02d}" for i in range(1, 28)]
session_id = [0, 1]
story_id = [0, 1, 2, 3]

sample_rate = 120
sample_t = 10
seq_length = 5

wav = wav_processor()

class Param:
    pass
param = Param()
param.C = 0.2
param.fps = 2
param.sr = sample_rate
b2e = Brain2Event(param)

base_dir = r"E:\NIPS2026\datasets\gwilliams2022"
seq_dir = rf'{base_dir}\seq'
label_dir = rf'{base_dir}\labels'
event_dir = rf'{base_dir}\events'
text_dir = rf'{base_dir}\texts'


def main():
    for subject in subjects_id:
        print(f"Processing subject {subject}")
        for session in session_id:
            print(f"+++ session {session}")
            for story in story_id:
                print(f"++++++++ story {story}")
                eeg_file = os.path.join(base_dir, fr"{subject}_session{session}_story{story}\meg-sr120-hp0-raw.fif")
                event_file = os.path.join(base_dir, fr"{subject}_session{session}_story{story}\events.csv")
                if not os.path.isfile(eeg_file):
                    continue

                os.makedirs(rf"{seq_dir}\{subject}_session{session}_story{story}", exist_ok=True)
                os.makedirs(rf"{label_dir}\{subject}_session{session}_story{story}", exist_ok=True)
                os.makedirs(rf"{event_dir}\{subject}_session{session}_story{story}", exist_ok=True)
                os.makedirs(rf"{text_dir}\{subject}_session{session}_story{story}", exist_ok=True)

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
                        torch.save(eegs, rf"{seq_dir}\{subject}_session{session}_story{story}\{num}.pth")
                        torch.save(labels, rf"{label_dir}\{subject}_session{session}_story{story}\{num}.pth")
                        torch.save(events, rf"{event_dir}\{subject}_session{session}_story{story}\{num}.pth")
                        torch.save(texts, rf"{text_dir}\{subject}_session{session}_story{story}\{num}.pth")
                        eegs, labels, texts = [], [], []
                        num += 1


if __name__ == '__main__':
    main()