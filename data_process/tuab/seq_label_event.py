import os
import torch
import pickle
from models.utils import Brain2Event
from tqdm import tqdm


class Param:
    pass
param = Param()
param.C = 0.2
param.fps = 2
param.sr = 200
b2e = Brain2Event(param)
seq_length = 5

root = r"yourpath\datasets\TUAB\edf"
eeg_dir = r"yourpath\datasets\TUAB\edf\processed_eeg"
pair_dir = r"yourpath\datasets\TUAB\edf\processed_pair"

modes = ['train', 'val', 'test']
for mode in modes:
    seq_dir = rf'{pair_dir}\{mode}\seq'
    label_dir = rf'{pair_dir}\{mode}\labels'
    event_dir = rf'{pair_dir}\{mode}\events'
    os.makedirs(seq_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(event_dir, exist_ok=True)

    eegs = []
    labels = []
    num = 0
    for file in tqdm(os.listdir(rf'{eeg_dir}\{mode}')):
        eeg_label = pickle.load(open(rf'{eeg_dir}\{mode}\{file}', 'rb'))
        eeg, label = torch.tensor(eeg_label['X']), eeg_label['y']

        eegs.append(eeg)
        labels.append(label)
        if len(eegs) == seq_length:
            eegs = torch.stack(eegs)
            labels = torch.tensor(labels)
            events = b2e.forward(eegs)
            torch.save(eegs, rf"{seq_dir}\{num}.pth")
            torch.save(labels, rf"{label_dir}\{num}.pth")
            torch.save(events, rf"{event_dir}\{num}.pth")
            eegs, labels = [], []
            num += 1
