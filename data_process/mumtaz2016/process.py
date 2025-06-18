import os
import mne
import numpy as np
import torch
import pickle
from models.utils import Brain2Event


def iter_files(rootDir):
    files_H, files_MDD = [], []
    for file in os.listdir(rootDir):
        if 'TASK' not in file:
            if 'MDD' in file:
                files_MDD.append(file)
            elif 'H' in file:
                files_H.append(file)
    return files_H, files_MDD


selected_channels = ['EEG Fp1-LE', 'EEG Fp2-LE', 'EEG F3-LE', 'EEG F4-LE', 'EEG C3-LE', 'EEG C4-LE', 'EEG P3-LE',
                     'EEG P4-LE', 'EEG O1-LE', 'EEG O2-LE', 'EEG F7-LE', 'EEG F8-LE', 'EEG T3-LE', 'EEG T4-LE',
                     'EEG T5-LE', 'EEG T6-LE', 'EEG Fz-LE', 'EEG Cz-LE', 'EEG Pz-LE']
rootDir = r'E:\NIPS2026\datasets\Mumtaz2016'
files_H, files_MDD = iter_files(rootDir)
files_H = sorted(files_H)
files_MDD = sorted(files_MDD)
print(files_H)
print(files_MDD)
print(len(files_H), len(files_MDD))


files_dict = {
    'train':[],
    'val':[],
    'test':[],
}

dataset = {
    'train': list(),
    'val': list(),
    'test': list(),
}

files_dict['train'].extend(files_H[:40])
files_dict['train'].extend(files_MDD[:42])
files_dict['test'].extend(files_H[40:48])
files_dict['test'].extend(files_MDD[42:52])
files_dict['val'].extend(files_H[48:])
files_dict['val'].extend(files_MDD[52:])

print(files_dict['train'])
print(files_dict['val'])
print(files_dict['test'])


class Param:
    pass
param = Param()
param.C = 0.2
param.fps = 10
param.sr = 200
b2e = Brain2Event(param)

for files_key in files_dict.keys():
    seq_dir = rf'E:\NIPS2026\datasets\Mumtaz2016\{files_key}\seq'
    label_dir = rf'E:\NIPS2026\datasets\Mumtaz2016\{files_key}\labels'
    event_dir = rf'E:\NIPS2026\datasets\Mumtaz2016\{files_key}\events'
    for file in files_dict[files_key]:
        raw = mne.io.read_raw_edf(os.path.join(rootDir, file), preload=True)
        # print(raw.info['ch_names'])
        raw.pick_channels(selected_channels, ordered=True)
        # print(raw.info['ch_names'])
        raw.resample(200)
        raw.filter(l_freq=0.3, h_freq=30)
        raw.notch_filter((50))
        # raw.plot_psd(average=True)
        eeg_array = raw.to_data_frame().values
        # print(raw.info)
        eeg_array = eeg_array[:, 1:]
        points, chs = eeg_array.shape
        # print(eeg_array.shape)
        a = points % (5 * 600)
        # print(a)
        if a != 0:
            eeg_array = eeg_array[:-a, :]
        eeg_array = eeg_array.reshape(-1, 5, 600, chs)
        eeg_array = eeg_array.transpose(0, 1, 3, 2)   # N, seq_len, channel, t
        eeg_array = torch.tensor(eeg_array, dtype=torch.float)
        # print(eeg_array.shape)
        labels = 1 if 'MDD' in file else 0
        labels = torch.full(eeg_array.shape[:2], labels, dtype=torch.long)

        epochs_events = []
        for seq in eeg_array:
            events = b2e.forward(seq)
            epochs_events.append(events)
        epochs_events = torch.stack(epochs_events)

        # s_index = file.find('S')
        # s_part = file[s_index:]
        # subject_id = file.split()[:2]
        # subject_id = subject_id[0] + '_' + subject_id[1]
        subject_id = file.split('.')[0]

        os.makedirs(rf"{seq_dir}\{subject_id}", exist_ok=True)
        os.makedirs(rf"{label_dir}\{subject_id}", exist_ok=True)
        os.makedirs(rf"{event_dir}\{subject_id}", exist_ok=True)
        num = 0
        for eeg, label, event in zip(eeg_array, labels, epochs_events):
            torch.save(eeg, rf"{seq_dir}\{subject_id}\{num}.pth")
            torch.save(label, rf"{label_dir}\{subject_id}\{num}.pth")
            torch.save(event, rf"{event_dir}\{subject_id}\{num}.pth")
            num += 1