import pdb
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random


subjects_id = [
            'S01', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10',
            'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19',
            'S20', 'S21', 'S22', 'S23', 'S25', 'S38', 'S39', 'S40', 'S41',
            'S42', 'S43', 'S44', 'S45', 'S48',
        ]


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
        subject_id = subjects_id.index(self.seqs_labels_path_pair[idx][0].split('\\')[-2])
        seq_path = self.seqs_labels_path_pair[idx][0]
        label_path = self.seqs_labels_path_pair[idx][1]
        event_path = self.seqs_labels_path_pair[idx][2]
        seq = torch.load(seq_path, map_location='cpu', weights_only=False)
        label = torch.load(label_path, map_location='cpu', weights_only=False)
        event = torch.load(event_path, map_location='cpu', weights_only=False)
        return seq, label, event, subject_id

    def collate(self, batch):
        x_seq = torch.stack([x[0] for x in batch]).float()
        y_label = torch.stack([x[1] for x in batch]).float()
        z_event = torch.stack([x[2] for x in batch]).float()
        w_s = torch.tensor([x[3] for x in batch]).long().unsqueeze(1).repeat(1, x_seq.shape[1])
        return x_seq, y_label, z_event, w_s


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
                batch_size=10,
                collate_fn=val_set.collate,
                shuffle=False,
                drop_last=True,
            ),
            'test': DataLoader(
                test_set,
                batch_size=10,
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
        for subject in subjects_id:
            subject_dirs_seq.append(os.path.join(self.seqs_dir, f'{subject}'))
            subject_dirs_labels.append(os.path.join(self.labels_dir, f'{subject}'))
            subject_dirs_events.append(os.path.join(self.events_dir, f'{subject}'))

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

    def split_dataset(self, seqs_labels_path_pair, seed=42):
        seqs_flatten = []
        for pair in seqs_labels_path_pair:
            seqs_flatten.extend(pair)

        random.seed(seed)
        random.shuffle(seqs_flatten)

        total = len(seqs_flatten)
        n_val = int(total * 0.2)
        n_train = total - 2 * n_val

        train_split = seqs_flatten[:n_train]
        val_split = seqs_flatten[n_train:n_train + n_val]
        test_split = seqs_flatten[n_train + n_val:]
        return train_split, val_split, test_split