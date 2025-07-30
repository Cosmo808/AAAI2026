from tqdm import tqdm
import os
import torch
import mne
from models.utils import Brain2Event

useless_ch = ['M1', 'M2', 'VEO', 'HEO']
trials_of_sessions = {
    '1': {'start': [30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204],
          'end': [102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401, 2607, 2901, 3172, 3359]},

    '2': {'start': [30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966, 2186, 2333, 2490, 2741],
          'end': [267, 488, 614, 773, 967, 1059, 1331, 1622, 1777, 1908, 2153, 2302, 2428, 2709, 2817]},

    '3': {'start': [30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888],
          'end': [321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066]},
}
labels_of_sessions = {
    '1': [4, 1, 3, 2, 0, 4, 1, 3, 2, 0, 4, 1, 3, 2, 0, ],
    '2': [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0, ],
    '3': [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0, ],
}

root_dir = r'yourpath\datasets\SEED-V\EEG_raw'
files = [file for file in os.listdir(root_dir)]
files = sorted(files)
print(files)

trials_split = {
    'train': range(5),
    'val': range(5, 10),
    'test': range(10, 15),
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
param.fps = 10
param.sr = 200
b2e = Brain2Event(param)

for file in files:
    if '.cnt' not in file:
        continue
    raw = mne.io.read_raw_cnt(os.path.join(root_dir, file), preload=True)
    raw.drop_channels(useless_ch)
    raw.resample(200)
    raw.filter(l_freq=0.3, h_freq=75)
    data_matrix = raw.get_data(units='uV')
    session_index = file.split('_')[1]
    data_trials = [
        data_matrix[:,
        trials_of_sessions[session_index]['start'][j] * 200:trials_of_sessions[session_index]['end'][j] * 200]
        for j in range(15)]
    labels = labels_of_sessions[session_index]
    for mode in trials_split.keys():
        seq_dir = rf'yourpath\datasets\SEED-V\{mode}\seq'
        label_dir = rf'yourpath\datasets\SEED-V\{mode}\labels'
        event_dir = rf'yourpath\datasets\SEED-V\{mode}\events'
        num = 0
        for index in tqdm(trials_split[mode]):
            data = data_trials[index]
            label = labels[index]
            data = data.reshape(62, -1, 1, 200)
            data = data.transpose(1, 2, 0, 3)

            data = torch.tensor(data)
            epochs_events = []
            for seq in data:
                events = b2e.forward(seq)
                epochs_events.append(events)
            epochs_events = torch.stack(epochs_events)

            label_s = torch.ones(size=data.shape[:2]) * label

            subject_id = file.split('.')[0]
            os.makedirs(rf"{seq_dir}\{subject_id}", exist_ok=True)
            os.makedirs(rf"{label_dir}\{subject_id}", exist_ok=True)
            os.makedirs(rf"{event_dir}\{subject_id}", exist_ok=True)
            for eeg, label, event in zip(data, label_s, epochs_events):
                torch.save(eeg.clone(), rf"{seq_dir}\{subject_id}\{num}.pth")  # [1, 62, 200]
                torch.save(label.clone(), rf"{label_dir}\{subject_id}\{num}.pth")
                torch.save(event.clone(), rf"{event_dir}\{subject_id}\{num}.pth")  # [1, 10, 2, 62]
                num += 1

