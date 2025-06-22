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
        subject_id = int(self.seqs_labels_path_pair[idx][0].split('\\')[-2].split('_')[0][1:]) - 1
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
        subject_id = int(self.seqs_labels_path_pair[idx][0].split('\\')[-2].split('_')[0]) - 1
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
        data_loader = DataLoader(alldata, batch_size=10, collate_fn=alldata.collate, shuffle=False, drop_last=True)
        return data_loader

    def load_path(self):
        subjects_id = [f"{i:02d}" for i in range(1, 28)]
        session_id = [0, 1]
        story_id = [0, 1, 2, 3]
        
        seqs_labels_path_pair = []
        subject_dirs_seq = []
        subject_dirs_labels = []
        subject_dirs_events = []
        subject_dirs_texts = []
        for subject in subjects_id:
            for session in session_id:
                for story in story_id:
                    if not os.path.exists(os.path.join(self.seqs_dir, f'{subject}_session{session}_story{story}')):
                        continue
                    subject_dirs_seq.append(os.path.join(self.seqs_dir, f'{subject}_session{session}_story{story}'))
                    subject_dirs_labels.append(os.path.join(self.labels_dir, f'{subject}_session{session}_story{story}'))
                    subject_dirs_events.append(os.path.join(self.events_dir, f'{subject}_session{session}_story{story}'))
                    subject_dirs_texts.append(os.path.join(self.texts_dir, f'{subject}_session{session}_story{story}'))

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

        subject_num = [
            8, 8, 4, 8, 8, 8, 8, 8, 8, 8,
            8, 4, 8, 8, 8, 4, 8, 8, 8, 4,
            4, 8, 8, 8, 8, 8, 8
        ]
        assert sum(subject_num) == len(seqs_labels_path_pair), "Length mismatch!"

        train_pairs, val_pairs, test_pairs = [], [], []

        start_idx = 0
        for i, num in enumerate(subject_num):
            end_idx = start_idx + num
            samples = seqs_labels_path_pair[start_idx:end_idx]
            start_idx = end_idx

            selected = random.sample(samples, num)
            if num == 8:
                train, val, test = selected[:5], selected[5:7], selected[7:]
            elif num == 4:
                train, val, test = selected[:2], selected[2:3], selected[3:]

            train_pairs.append(train)
            val_pairs.append(val)
            test_pairs.append(test)

        train_pairs = [item for sublist in train_pairs for item in sublist]
        val_pairs = [item for sublist in val_pairs for item in sublist]
        test_pairs = [item for sublist in test_pairs for item in sublist]
        return train_pairs, val_pairs, test_pairs