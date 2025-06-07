import pdb
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
        subject_id = int(self.seqs_labels_path_pair[idx][0].split('\\')[-2][1]) - 1
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


class AllData(Dataset):
    def __init__(
            self,
            seqs_labels_path_pair
    ):
        super(AllData, self).__init__()
        self.seqs_labels_path_pair = []
        for s in seqs_labels_path_pair:
            self.seqs_labels_path_pair.extend(s)

    def __len__(self):
        return len((self.seqs_labels_path_pair))

    def __getitem__(self, idx):
        subject_id = int(self.seqs_labels_path_pair[idx][0].split('\\')[-2][1]) - 1
        seq_path = self.seqs_labels_path_pair[idx][0]
        label_path = self.seqs_labels_path_pair[idx][1]
        event_path = self.seqs_labels_path_pair[idx][2]
        text_path = self.seqs_labels_path_pair[idx][3]
        seq = torch.load(seq_path, map_location='cpu', weights_only=False)
        label = torch.load(label_path, map_location='cpu', weights_only=False)
        event = torch.load(event_path, map_location='cpu', weights_only=False)
        text = torch.load(text_path, map_location='cpu', weights_only=False)
        return seq, label, event, subject_id, text

    def collate(self, batch):
        x_seq = torch.stack([x[0] for x in batch]).float()
        y_label = torch.stack([x[1] for x in batch]).float()
        z_event = torch.stack([x[2] for x in batch]).float()
        w_s = torch.tensor([x[3] for x in batch]).long().unsqueeze(1).repeat(1, x_seq.shape[1])
        v_text = [x[4] for x in batch]
        return x_seq, y_label, z_event, w_s, v_text


class LoadDataset(object):
    def __init__(self, args):
        self.args = args
        datasets_dir = os.path.join(args.base_dir, 'datasets', args.datasets)
        self.seqs_dir = os.path.join(datasets_dir, 'seq')
        self.labels_dir = os.path.join(datasets_dir, 'labels')
        self.events_dir = os.path.join(datasets_dir, 'events')
        self.texts_dir = os.path.join(datasets_dir, 'texts')
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

    def get_alldata(self):
        alldata = AllData(self.seqs_labels_path_pair)
        data_loader = DataLoader(alldata, batch_size=10, collate_fn=alldata.collate, shuffle=False, drop_last=False)
        return data_loader

    def load_path(self):
        seqs_labels_path_pair = []
        subject_dirs_seq = []
        subject_dirs_labels = []
        subject_dirs_events = []
        subject_dirs_texts = []
        for subject in range(1, 20):
            for run in range(1, 21):
                subject_dirs_seq.append(os.path.join(self.seqs_dir, f's{subject}_r{run}'))
                subject_dirs_labels.append(os.path.join(self.labels_dir, f's{subject}_r{run}'))
                subject_dirs_events.append(os.path.join(self.events_dir, f's{subject}_r{run}'))
                subject_dirs_texts.append(os.path.join(self.texts_dir, f's{subject}_r{run}'))

        for subject_seq, subject_label, subject_event, subject_text in zip(subject_dirs_seq, subject_dirs_labels, subject_dirs_events, subject_dirs_texts):
            subject_pairs = []
            seq_fnames = os.listdir(subject_seq)
            label_fnames = os.listdir(subject_label)
            events_fnames = os.listdir(subject_event)
            texts_fnames = os.listdir(subject_text)
            for seq_fname, label_fname, events_fname, texts_fname in zip(seq_fnames, label_fnames, events_fnames, texts_fnames):
                subject_pairs.append((os.path.join(subject_seq, seq_fname),
                                      os.path.join(subject_label, label_fname),
                                      os.path.join(subject_event, events_fname),
                                      os.path.join(subject_text, texts_fname),))
            seqs_labels_path_pair.append(subject_pairs)
        return seqs_labels_path_pair

    def split_dataset(self, seqs_labels_path_pair, seed=42):
        random.seed(seed)
        random.shuffle(seqs_labels_path_pair)

        total = len(seqs_labels_path_pair)
        n_train = int(total * 0.6)
        n_val = int(total * 0.2)
        n_test = total - n_train - n_val  # ensure full coverage

        train_split = seqs_labels_path_pair[:n_train]
        val_split = seqs_labels_path_pair[n_train:n_train + n_val]
        test_split = seqs_labels_path_pair[n_train + n_val:]

        train_pairs, val_pairs, test_pairs = [], [], []
        for group in train_split:
            train_pairs.extend(group)
        for group in val_split:
            val_pairs.extend(group)
        for group in test_split:
            test_pairs.extend(group)
        return train_pairs, val_pairs, test_pairs