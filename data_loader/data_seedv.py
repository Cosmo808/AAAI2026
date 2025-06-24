import torch
from torch.utils.data import Dataset, DataLoader
import os
import torch


class CustomDataset(Dataset):
    def __init__(
            self,
            data_dir,
            mode,
    ):
        super(CustomDataset, self).__init__()
        self.dataset_dir = os.path.join(data_dir, mode)
        self.seqs_labels_path_pair = self.load_path()

    def __len__(self):
        return len(self.seqs_labels_path_pair)

    def __getitem__(self, idx):
        subject_id = int(self.seqs_labels_path_pair[idx][0].split('\\')[-2].split('_')[0]) - 1
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

    def load_path(self):
        seqs_labels_path_pair = []
        subject_dirs_seq = []
        subject_dirs_labels = []
        subject_dirs_events = []

        subject_ids = os.listdir(os.path.join(self.dataset_dir, 'events'))
        for subject_id in subject_ids:
            subject_dirs_seq.append(os.path.join(self.dataset_dir, 'seq', subject_id))
            subject_dirs_labels.append(os.path.join(self.dataset_dir, 'labels', subject_id))
            subject_dirs_events.append(os.path.join(self.dataset_dir, 'events', subject_id))

        for subject_seq, subject_label, subject_event in zip(subject_dirs_seq, subject_dirs_labels, subject_dirs_events):
            # subject_pairs = []
            seq_fnames = os.listdir(subject_seq)
            label_fnames = os.listdir(subject_label)
            events_fnames = os.listdir(subject_event)
            for seq_fname, label_fname, events_fname in zip(seq_fnames, label_fnames, events_fnames):
                subject_pairs = [os.path.join(subject_seq, seq_fname),
                                 os.path.join(subject_label, label_fname),
                                 os.path.join(subject_event, events_fname)]
                seqs_labels_path_pair.append(subject_pairs)
        return seqs_labels_path_pair


class LoadDataset(object):
    def __init__(self, args):
        self.args = args
        self.datasets_dir = os.path.join(args.base_dir, 'datasets', args.datasets, 'seq_5')

    def get_data_loader(self):
        train_set = CustomDataset(self.datasets_dir, mode='train')
        val_set = CustomDataset(self.datasets_dir, mode='val')
        test_set = CustomDataset(self.datasets_dir, mode='test')
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
