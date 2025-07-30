import scipy
from scipy import signal
import os
import torch
from models.utils import Brain2Event
from tqdm import tqdm


root_dir = r'yourpath\datasets\SHU-MI\mat'
files = [file for file in os.listdir(root_dir)]
files = sorted(files)
# print(files)

files_dict = {
    'train':files[:75],
    'val':files[75:100],
    'test':files[100:],
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
param.fps = 5
param.sr = 200
b2e = Brain2Event(param)

for files_key in files_dict.keys():
    seq_dir = rf'yourpath\datasets\SHU-MI\{files_key}\seq'
    label_dir = rf'yourpath\datasets\SHU-MI\{files_key}\labels'
    event_dir = rf'yourpath\datasets\SHU-MI\{files_key}\events'
    for file in tqdm(files_dict[files_key]):
        data = scipy.io.loadmat(os.path.join(root_dir, file))
        eeg = data['data']
        bz, ch_num, points = eeg.shape
        eeg_resample = signal.resample(eeg, 800, axis=2)
        eeg_resample = eeg_resample[:bz // 5 * 5]
        eeg_ = torch.tensor(eeg_resample.reshape(bz // 5, 5, ch_num, 800))

        labels = data['labels'][0]
        labels = torch.tensor(labels[:bz // 5 * 5].reshape(bz // 5, 5)) - 1

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
        for eeg, label, event in zip(eeg_, labels, epochs_events):
            torch.save(eeg.clone(), rf"{seq_dir}\{subject_id}\{num}.pth")  # [5, 32, 800]
            torch.save(label.clone(), rf"{label_dir}\{subject_id}\{num}.pth")
            torch.save(event.clone(), rf"{event_dir}\{subject_id}\{num}.pth")  # [5, 20, 2, 32]
            num += 1
