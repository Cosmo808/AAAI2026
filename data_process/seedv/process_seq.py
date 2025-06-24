from tqdm import tqdm
import os
import torch


seq_len = 5
root_dir = r"E:\NIPS2026\datasets\SEED-V"
modes = ['train', 'val', 'test']

for mode in modes:
    seq_dir = rf'E:\NIPS2026\datasets\SEED-V\{mode}\seq'
    label_dir = rf'E:\NIPS2026\datasets\SEED-V\{mode}\labels'
    event_dir = rf'E:\NIPS2026\datasets\SEED-V\{mode}\events'

    subject_ids = os.listdir(seq_dir)
    for subject_id in tqdm(subject_ids):
        seq_files = os.listdir(os.path.join(seq_dir, subject_id))
        total_len = len(seq_files)

        for i in range(total_len // seq_len):
            seqs, labels, events = [], [], []
            for j in range(i * seq_len, (i + 1) * seq_len):
                seqs.append(torch.load(os.path.join(seq_dir, subject_id, f'{j}.pth'), weights_only=False, map_location='cpu'))
                labels.append(torch.load(os.path.join(label_dir, subject_id, f'{j}.pth'), weights_only=False, map_location='cpu'))
                events.append(torch.load(os.path.join(event_dir, subject_id, f'{j}.pth'), weights_only=False, map_location='cpu'))
            seqs = torch.cat(seqs, dim=0)
            labels = torch.cat(labels, dim=0)
            events = torch.cat(events, dim=0)

            new_seq_dir = rf'E:\NIPS2026\datasets\SEED-V\seq_{seq_len}\{mode}\seq\{subject_id}'
            new_label_dir = rf'E:\NIPS2026\datasets\SEED-V\seq_{seq_len}\{mode}\labels\{subject_id}'
            new_event_dir = rf'E:\NIPS2026\datasets\SEED-V\seq_{seq_len}\{mode}\events\{subject_id}'
            os.makedirs(new_seq_dir, exist_ok=True)
            os.makedirs(new_label_dir, exist_ok=True)
            os.makedirs(new_event_dir, exist_ok=True)

            torch.save(seqs.clone(), rf"{new_seq_dir}\{i}.pth")  # [5, 62, 200]
            torch.save(labels.clone(), rf"{new_label_dir}\{i}.pth")
            torch.save(events.clone(), rf"{new_event_dir}\{i}.pth")  # [5, 10, 2, 62]
