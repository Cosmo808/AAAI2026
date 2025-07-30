from scipy import signal
import os
import pickle
import numpy as np
from models.utils import Brain2Event
from tqdm import tqdm
import torch

labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8])
root_dir = r'yourpath\datasets\FACED\Processed_data'
files = [file for file in os.listdir(root_dir)]
files = sorted(files)

files_dict = {
    'train': files[:80],
    'val': files[80:100],
    'test': files[100:],
}

dataset = {
    'train': list(),
    'val': list(),
    'test': list(),
}

class Param:
    pass
param = Param()
param.C = 0.2
param.fps = 1
param.sr = 200
b2e = Brain2Event(param)

for files_key in files_dict.keys():
    seq_dir = rf'yourpath\datasets\FACED\{files_key}\seq'
    label_dir = rf'yourpath\datasets\FACED\{files_key}\labels'
    event_dir = rf'yourpath\datasets\FACED\{files_key}\events'
    for file in tqdm(files_dict[files_key]):
        f = open(os.path.join(root_dir, file), 'rb')
        array = pickle.load(f)
        eeg = signal.resample(array, 6000, axis=2)
        eeg_ = eeg.reshape(28, 32, 30, 200)
        eeg_ = torch.tensor(eeg_).view(7, 4, 32, 6000)
        labels_ = torch.tensor(labels).view(7, 4)

        epochs_events = []
        for seq in eeg_:
            events = b2e.forward(seq)
            epochs_events.append(events)
        epochs_events = torch.stack(epochs_events)

        subject_id = file.split('.')[0]
        os.makedirs(rf"{seq_dir}\{subject_id}", exist_ok=True)
        os.makedirs(rf"{label_dir}\{subject_id}", exist_ok=True)
        os.makedirs(rf"{event_dir}\{subject_id}", exist_ok=True)
        num = 0
        for eeg, label, event in zip(eeg_, labels_, epochs_events):
            torch.save(eeg.clone(), rf"{seq_dir}\{subject_id}\{num}.pth")  # [4, 32, 6000]
            torch.save(label.clone(), rf"{label_dir}\{subject_id}\{num}.pth")
            torch.save(event.clone(), rf"{event_dir}\{subject_id}\{num}.pth")  # [4, 30, 2, 32]
            num += 1

