import h5py
import scipy
import torch
from scipy import signal
import os
from models.utils import Brain2Event
import numpy as np
import pandas as pd
from tqdm import tqdm

train_dir = r'E:\NIPS2026\datasets\BCIC2020\Training set'
val_dir = r'E:\NIPS2026\datasets\BCIC2020\Validation set'
test_dir = r'E:\NIPS2026\datasets\BCIC2020\Test set'

files_dict = {
    'train': sorted([file for file in os.listdir(train_dir)]),
    'val': sorted([file for file in os.listdir(val_dir)]),
    'test': sorted([file for file in os.listdir(test_dir)]),
}

print(files_dict)

dataset = {
    'train': list(),
    'val': list(),
    'test': list(),
}

class Param:
    pass
param = Param()
param.C = 0.2
param.fps = 10
param.sr = 200
b2e = Brain2Event(param)
seq_length = 5

seq_dir = r'E:\NIPS2026\datasets\BCIC2020\train\seq'
label_dir = r'E:\NIPS2026\datasets\BCIC2020\train\labels'
event_dir = r'E:\NIPS2026\datasets\BCIC2020\train\events'
for file in tqdm(files_dict['train']):
    subject_id = file.split('.')[0]
    os.makedirs(rf"{seq_dir}\{subject_id}", exist_ok=True)
    os.makedirs(rf"{label_dir}\{subject_id}", exist_ok=True)
    os.makedirs(rf"{event_dir}\{subject_id}", exist_ok=True)

    data = scipy.io.loadmat(os.path.join(train_dir, file))
    # print(data['epo_train'][0][0][0])
    eeg = data['epo_train'][0][0][4].transpose(2, 1, 0)
    labels = data['epo_train'][0][0][5].transpose(1, 0)
    eeg = eeg[:, :, -768:]
    labels = np.argmax(labels, axis=1)
    eeg = signal.resample(eeg, 600, axis=2)
    for i in range(eeg.shape[0] // 5):
        eeg_seq = torch.tensor(eeg[i * 5:(i + 1) * 5])
        label_seq = torch.tensor(labels[i * 5:(i + 1) * 5])
        event_seq = b2e.forward(eeg_seq)
        torch.save(eeg_seq, rf"{seq_dir}\{subject_id}\{i}.pth")
        torch.save(label_seq, rf"{label_dir}\{subject_id}\{i}.pth")
        torch.save(event_seq, rf"{event_dir}\{subject_id}\{i}.pth")

seq_dir = r'E:\NIPS2026\datasets\BCIC2020\val\seq'
label_dir = r'E:\NIPS2026\datasets\BCIC2020\val\labels'
event_dir = r'E:\NIPS2026\datasets\BCIC2020\val\events'
for file in tqdm(files_dict['val']):
    subject_id = file.split('.')[0]
    os.makedirs(rf"{seq_dir}\{subject_id}", exist_ok=True)
    os.makedirs(rf"{label_dir}\{subject_id}", exist_ok=True)
    os.makedirs(rf"{event_dir}\{subject_id}", exist_ok=True)

    data = scipy.io.loadmat(os.path.join(val_dir, file))
    eeg = data['epo_validation'][0][0][4].transpose(2, 1, 0)
    labels = data['epo_validation'][0][0][5].transpose(1, 0)
    eeg = eeg[:, :, -768:]
    labels = np.argmax(labels, axis=1)
    eeg = signal.resample(eeg, 600, axis=2)
    for i in range(eeg.shape[0] // 5):
        eeg_seq = torch.tensor(eeg[i * 5:(i + 1) * 5])
        label_seq = torch.tensor(labels[i * 5:(i + 1) * 5])
        event_seq = b2e.forward(eeg_seq)
        torch.save(eeg_seq, rf"{seq_dir}\{subject_id}\{i}.pth")
        torch.save(label_seq, rf"{label_dir}\{subject_id}\{i}.pth")
        torch.save(event_seq, rf"{event_dir}\{subject_id}\{i}.pth")

df = pd.read_excel(r"E:\NIPS2026\datasets\BCIC2020\Track3_Answer Sheet_Test.xlsx")
df_ = df.head(53)
all_labels = df_.values
all_labels = all_labels[2:, 1:][:, 1:30:2].transpose(1, 0)

seq_dir = r'E:\NIPS2026\datasets\BCIC2020\test\seq'
label_dir = r'E:\NIPS2026\datasets\BCIC2020\test\labels'
event_dir = r'E:\NIPS2026\datasets\BCIC2020\test\events'
for labels, file in tqdm(zip(all_labels, files_dict['test'])):
    subject_id = file.split('.')[0]
    os.makedirs(rf"{seq_dir}\{subject_id}", exist_ok=True)
    os.makedirs(rf"{label_dir}\{subject_id}", exist_ok=True)
    os.makedirs(rf"{event_dir}\{subject_id}", exist_ok=True)

    data = h5py.File(os.path.join(test_dir, file))
    eeg = data['epo_test']['x'][:]
    eeg = eeg[:, :, -768:]
    eeg = signal.resample(eeg, 600, axis=2)
    for i in range(eeg.shape[0] // 5):
        eeg_seq = torch.tensor(eeg[i * 5:(i + 1) * 5])
        label_seq = torch.tensor(list(labels[i * 5:(i + 1) * 5])) - 1
        event_seq = b2e.forward(eeg_seq)
        torch.save(eeg_seq, rf"{seq_dir}\{subject_id}\{i}.pth")
        torch.save(label_seq, rf"{label_dir}\{subject_id}\{i}.pth")
        torch.save(event_seq, rf"{event_dir}\{subject_id}\{i}.pth")


