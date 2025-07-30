import os
import mne
from models.utils import Brain2Event
import torch
from tqdm import tqdm

root_dir = r'yourpath\datasets\MentalArithmetic'
files = [file for file in os.listdir(root_dir)]
files = sorted(files)
print(files)

files_dict = {
    'train': files[:56],
    'val': files[56:64],
    'test': files[64:],
}
print(files_dict)
dataset = {
    'train': list(),
    'val': list(),
    'test': list(),
}

selected_channels = ['EEG Fp1', 'EEG Fp2', 'EEG F3', 'EEG F4', 'EEG F7', 'EEG F8', 'EEG T3', 'EEG T4',
                     'EEG C3', 'EEG C4', 'EEG T5', 'EEG T6', 'EEG P3', 'EEG P4', 'EEG O1', 'EEG O2',
                     'EEG Fz', 'EEG Cz', 'EEG Pz', 'EEG A2-A1']


class Param:
    pass
param = Param()
param.C = 0.2
param.fps = 10
param.sr = 200
b2e = Brain2Event(param)

for files_key in files_dict.keys():
    seq_dir = rf'yourpath\datasets\MentalArithmetic\{files_key}\seq'
    label_dir = rf'yourpath\datasets\MentalArithmetic\{files_key}\labels'
    event_dir = rf'yourpath\datasets\MentalArithmetic\{files_key}\events'
    for file in tqdm(files_dict[files_key]):
        if '.edf' not in file:
            continue
        raw = mne.io.read_raw_edf(os.path.join(root_dir, file), preload=True)
        raw.pick(selected_channels)
        raw.reorder_channels(selected_channels)
        raw.resample(200)

        eeg = raw.get_data(units='uV')
        chs, points = eeg.shape
        a = points % (5 * 600)
        if a != 0:
            eeg = eeg[:, :-a]
        eeg = eeg.reshape(20, -1, 5, 600).transpose(1, 2, 0, 3)  # N, seq_len, chs, t; N, 5, 20, 600
        eeg_array = torch.tensor(eeg, dtype=torch.float)
        label_s = int(file[-5]) - 1
        labels = torch.full(eeg_array.shape[:2], label_s, dtype=torch.long)
        subject_id = file.split('_')[0]

        epochs_events = []
        for seq in eeg_array:
            events = b2e.forward(seq)
            epochs_events.append(events)
        epochs_events = torch.stack(epochs_events)

        os.makedirs(rf"{seq_dir}\{subject_id}", exist_ok=True)
        os.makedirs(rf"{label_dir}\{subject_id}", exist_ok=True)
        os.makedirs(rf"{event_dir}\{subject_id}", exist_ok=True)
        num = 0
        for eeg, label, event in zip(eeg_array, labels, epochs_events):
            torch.save(eeg.clone(), rf"{seq_dir}\{subject_id}\{num}_{label_s}.pth")
            torch.save(label.clone(), rf"{label_dir}\{subject_id}\{num}_{label_s}.pth")
            torch.save(event.clone(), rf"{event_dir}\{subject_id}\{num}_{label_s}.pth")
            num += 1
