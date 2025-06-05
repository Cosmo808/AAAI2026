# %%
import torch
from mne.io import concatenate_raws
from edf_ import read_raw_edf
import matplotlib.pyplot as plt
import mne
import os
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
from sklearn.preprocessing import StandardScaler
from models.utils import Brain2Event


class Param:
    pass
param = Param()
param.C = 0.2
param.fps = 1
param.sr = 200
b2e = Brain2Event(param)

dir_path = r'E:\NIPS2026\datasets\ISRUC'

seq_dir = r'E:\NIPS2026\datasets\ISRUC\seq'
label_dir = r'E:\NIPS2026\datasets\ISRUC\labels'
event_dir = r'E:\NIPS2026\datasets\ISRUC\events'

psg_f_names = []
label_f_names = []
for i in range(1, 101):
    numstr = str(i)
    psg_f_names.append(f'{dir_path}/{numstr}/{numstr}.rec')
    label_f_names.append(f'{dir_path}/{numstr}/{numstr}_1.txt')

# psg_f_names.sort()
# label_f_names.sort()

psg_label_f_pairs = []
for psg_f_name, label_f_name in zip(psg_f_names, label_f_names):
    if psg_f_name[:-4] == label_f_name[:-6]:
        psg_label_f_pairs.append((psg_f_name, label_f_name))
for item in psg_label_f_pairs:
    print(item)

label2id = {'0': 0,
            '1': 1,
            '2': 2,
            '3': 3,
            '5': 4,}
print(label2id)
# %%
# signal_name = ['LOC-A2', 'F4-A1']
n = 0
num_seqs = 0
num_labels = 0
num_events = 0
for psg_f_name, label_f_name in tqdm(psg_label_f_pairs):
    n += 1
    labels_list = []

    raw = read_raw_edf(os.path.join(dir_path, psg_f_name), preload=True)
    # raw.pick_channels(signal_name)
    # raw.resample(sfreq=200)
    raw.filter(0.3, 35, fir_design='firwin')
    raw.notch_filter((50))
    if n == 1:
        print(raw.info)

    psg_array = raw.to_data_frame().values
    psg_array = psg_array[:, 1:]
    psg_array = psg_array[:, 2:8]

    i = psg_array.shape[0] % (30 * 200)
    if i > 0:
        psg_array = psg_array[:-i, :]
    psg_array = psg_array.reshape(-1, 30 * 200, 6)

    a = psg_array.shape[0] % 20
    if a > 0:
        psg_array = psg_array[:-a, :, :]
    psg_array = psg_array.reshape(-1, 20, 30 * 200, 6)
    epochs_seq = psg_array.transpose(0, 1, 3, 2)
    # print(epochs_seq.shape)

    for line in open(os.path.join(dir_path, label_f_name)).readlines():
        line_str = line.strip()
        if line_str != '':
            labels_list.append(label2id[line_str])
    labels_array = np.array(labels_list)
    if a > 0:
        labels_array = labels_array[:-a]
    labels_seq = labels_array.reshape(-1, 20)
    # print(labels_seq.shape)

    epochs_events = []
    for seq in epochs_seq:
        seq = torch.tensor(seq)
        events = b2e.forward(seq)
        epochs_events.append(events)
    epochs_events = torch.stack(epochs_events)

    if not os.path.isdir(rf'{seq_dir}/ISRUC-group1-{str(n)}'):
        os.makedirs(rf'{seq_dir}/ISRUC-group1-{str(n)}')
    for seq in epochs_seq:
        seq_name = rf'{seq_dir}/ISRUC-group1-{str(n)}/ISRUC-group1-{str(n)}-{str(num_seqs)}.npy'
        with open(seq_name, 'wb') as f:
            np.save(f, seq)
        num_seqs += 1

    if not os.path.isdir(rf'{label_dir}/ISRUC-group1-{str(n)}'):
        os.makedirs(rf'{label_dir}/ISRUC-group1-{str(n)}')
    for label in labels_seq:
        label_name = rf'{label_dir}/ISRUC-group1-{str(n)}/ISRUC-group1-{str(n)}-{str(num_labels)}.npy'
        with open(label_name, 'wb') as f:
            np.save(f, label)
        num_labels += 1

    if not os.path.isdir(rf'{event_dir}/ISRUC-group1-{str(n)}'):
        os.makedirs(rf'{event_dir}/ISRUC-group1-{str(n)}')
    for events in epochs_events:
        events_name = rf'{event_dir}/ISRUC-group1-{str(n)}/ISRUC-group1-{str(n)}-{str(num_events)}.pth'
        torch.save(events, events_name)
        num_events += 1


