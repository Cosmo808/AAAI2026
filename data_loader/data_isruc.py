import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random


class CustomDataset(Dataset):
    def __init__(
            self,
            seqs_labels_path_pair
    ):
        super(CustomDataset, self).__init__()
        self.seqs_labels_path_pair = seqs_labels_path_pair

    def __len__(self):
        return len((self.seqs_labels_path_pair))

    def __getitem__(self, idx):
        seq_path = self.seqs_labels_path_pair[idx][0]
        label_path = self.seqs_labels_path_pair[idx][1]
        event_path = self.seqs_labels_path_pair[idx][2]
        seq = np.load(seq_path)
        label = np.load(label_path)
        event = torch.load(event_path, map_location='cpu', weights_only=False)
        return seq, label, event

    def collate(self, batch):
        x_seq = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        z_event = np.array([x[2] for x in batch])
        return to_tensor(x_seq), to_tensor(y_label).long(), to_tensor(z_event).float()


class LoadDataset(object):
    def __init__(self, args):
        self.args = args
        datasets_dir = os.path.join(args.base_dir, 'datasets', args.datasets)
        self.seqs_dir = os.path.join(datasets_dir, 'seq')
        self.labels_dir = os.path.join(datasets_dir, 'labels')
        self.events_dir = os.path.join(datasets_dir, 'events')
        self.seqs_labels_path_pair = self.load_path()

    def get_data_loader(self):
        train_pairs, val_pairs, test_pairs = self.split_dataset(self.seqs_labels_path_pair)
        train_set = CustomDataset(train_pairs)
        val_set = CustomDataset(val_pairs)
        test_set = CustomDataset(test_pairs)
        print("Sample Size:")
        print("---Total:", len(train_set) + len(val_set) + len(test_set))
        print("---Train/Val/Test:", len(train_set), len(val_set), len(test_set))
        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.args.bs,
                collate_fn=train_set.collate,
                shuffle=True,
                drop_last=True,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.args.bs,
                collate_fn=val_set.collate,
                shuffle=False,
                drop_last=True,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.args.bs,
                collate_fn=test_set.collate,
                shuffle=False,
                drop_last=True,
            ),
        }
        return data_loader

    def load_path(self):
        seqs_labels_path_pair = []
        subject_dirs_seq = []
        subject_dirs_labels = []
        subject_dirs_events = []
        for subject_num in range(1, 101):
            subject_dirs_seq.append(os.path.join(self.seqs_dir, f'ISRUC-group1-{subject_num}'))
            subject_dirs_labels.append(os.path.join(self.labels_dir, f'ISRUC-group1-{subject_num}'))
            subject_dirs_events.append(os.path.join(self.events_dir, f'ISRUC-group1-{subject_num}'))

        for subject_seq, subject_label, subject_event in zip(subject_dirs_seq, subject_dirs_labels, subject_dirs_events):
            subject_pairs = []
            seq_fnames = os.listdir(subject_seq)
            label_fnames = os.listdir(subject_label)
            events_fnames = os.listdir(subject_event)
            for seq_fname, label_fname, events_fname in zip(seq_fnames, label_fnames, events_fnames):
                subject_pairs.append((os.path.join(subject_seq, seq_fname),
                                      os.path.join(subject_label, label_fname),
                                      os.path.join(subject_event, events_fname)))
            seqs_labels_path_pair.append(subject_pairs)
        return seqs_labels_path_pair

    def split_dataset(self, seqs_labels_path_pair):
        train_pairs = []
        val_pairs = []
        test_pairs = []

        for i in range(100):
            if i < 80:
                train_pairs.extend(seqs_labels_path_pair[i])
            elif i < 90:
                val_pairs.extend(seqs_labels_path_pair[i])
            else:
                test_pairs.extend(seqs_labels_path_pair[i])
        # print(train_pairs, val_pairs, test_pairs)
        return train_pairs, val_pairs, test_pairs


def to_tensor(array):
    return torch.from_numpy(array).float()