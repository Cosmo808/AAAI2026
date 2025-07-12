import os
import mne
import torch
from models.utils import Brain2Event
from tqdm import tqdm

tasks = ['04', '06', '08', '10', '12', '14'] # select the data for motor imagery

root_dir = r'E:\NIPS2026\datasets\PhysioNet-MI\files\eegmmidb\1.0.0'
files = [file for file in os.listdir(root_dir)]
files = sorted(files)

files_dict = {
    'train': files[:70],
    'val': files[70:89],
    'test': files[89:109],
}

print(files_dict)

dataset = {
    'train': list(),
    'val': list(),
    'test': list(),
}

selected_channels = ['Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..',
                     'C4..', 'C6..', 'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.', 'Fp2.',
                     'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..',
                     'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..', 'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..',
                     'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..', 'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.',
                     'O1..', 'Oz..', 'O2..', 'Iz..']

class Param:
    pass
param = Param()
param.C = 0.2
param.fps = 5
param.sr = 200
b2e = Brain2Event(param)

for files_key in files_dict.keys():
    seq_dir = rf'E:\NIPS2026\datasets\PhysioNet-MI\{files_key}\seq'
    label_dir = rf'E:\NIPS2026\datasets\PhysioNet-MI\{files_key}\labels'
    event_dir = rf'E:\NIPS2026\datasets\PhysioNet-MI\{files_key}\events'
    for file in tqdm(files_dict[files_key]):
        for task in tasks:
            try:
                raw = mne.io.read_raw_edf(os.path.join(root_dir, file, f'{file}R{task}.edf'), preload=True)
            except FileNotFoundError:
                continue
            raw.pick_channels(selected_channels, ordered=True)
            if len(raw.info['bads']) > 0:
                # print('interpolate_bads')
                raw.interpolate_bads()
            raw.set_eeg_reference(ref_channels='average')
            raw.filter(l_freq=0.3, h_freq=None)
            raw.notch_filter((60))
            raw.resample(200)
            events_from_annot, event_dict = mne.events_from_annotations(raw)
            epochs = mne.Epochs(raw,
                                events_from_annot,
                                event_dict,
                                tmin=0,
                                tmax=4. - 1.0 / raw.info['sfreq'],
                                baseline=None,
                                preload=True)
            data = epochs.get_data(units='uV')
            events = epochs.events[:, 2]
            data = data[:, :, -800:]

            eegs, labels = [], []
            for eeg, label in zip(data, events):
                if label != 1:
                    eegs.append(torch.tensor(eeg))
                    labels.append(label - 2 if task in ['04', '08', '12'] else label)

            eegs = torch.stack(eegs)
            labels = torch.tensor(labels)

            bz, ch_nums, _ = eegs.shape
            eegs = eegs[:bz // 5 * 5].view(bz // 5, 5, ch_nums, 800)
            labels = labels[:bz // 5 * 5].view(bz // 5, 5)

            epochs_events = []
            for seq in eegs:
                events = b2e.forward(seq)
                epochs_events.append(events)
            epochs_events = torch.stack(epochs_events)

            subject_id = f'{file}R{task}'
            os.makedirs(rf"{seq_dir}\{subject_id}", exist_ok=True)
            os.makedirs(rf"{label_dir}\{subject_id}", exist_ok=True)
            os.makedirs(rf"{event_dir}\{subject_id}", exist_ok=True)
            num = 0
            for eeg, label, event in zip(eegs, labels, epochs_events):
                torch.save(eeg.clone(), rf"{seq_dir}\{subject_id}\{num}.pth")  # [5, 64, 800]
                torch.save(label.clone(), rf"{label_dir}\{subject_id}\{num}.pth")
                torch.save(event.clone(), rf"{event_dir}\{subject_id}\{num}.pth")  # [5, 20, 2, 64]
                num += 1