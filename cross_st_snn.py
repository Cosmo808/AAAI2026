import argparse
import pdb
import random
import torch
import matplotlib.pyplot as plt
from einops import rearrange

from models.snn import SAS
from models.utils import *

from data_loader import data_isruc, data_broderick2019, data_brennan2019, data_mumtaz2016, data_mental, data_shumi, \
    data_tuab, data_tuev, data_bcic2020, data_schoffelen2019, data_gwilliams2022, data_seedvig, data_seedv, data_faced, data_physio

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default=r"E:\NIPS2026")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--early_stop_epoch', type=int, default=20)
    parser.add_argument('--bs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--multi_lr', action='store_true', default=False)
    parser.add_argument('--weight_decay', type=float, default=5e-2)
    parser.add_argument('--grad_clip', type=float, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    parser.add_argument('--datasets', type=str, default='TUEV',
                        choices=['brennan2019', 'broderick2019', 'schoffelen2019', 'gwilliams2022',
                                 'SEED-VIG',
                                 'ISRUC', 'TUEV', 'BCIC2020', 'SEED-V', 'FACED', 'PhysioNet-MI',
                                 'Mumtaz2016', 'MentalArithmetic', 'TUAB', 'SHU-MI'])
    parser.add_argument('--model', type=str, default='cbramod', choices=['simplecnn', 'cbramod', 'labram'])
    parser.add_argument('--n_negatives', type=int, default=None)
    parser.add_argument('--n_subjects', type=int, default=1)
    parser.add_argument('--n_channels', type=int, default=None)
    parser.add_argument('--n_slice', type=int, default=1)
    parser.add_argument('--sr', type=int, default=None)
    parser.add_argument('--fps', type=int, default=1)
    parser.add_argument('--C', type=float, default=0.2)

    parser.add_argument('--ckpt_snn', type=str, default=None)
    parser.add_argument('--ckpt_ann', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=r"E:\NIPS2026\ckpt")
    parser.add_argument('--load_lbm', action='store_true', default=False)
    parser.add_argument('--foundation_dir', type=str, default=r"E:\NIPS2026\ckpt\cbramod-base.pth")
    parser.add_argument('--frozen_ann', action='store_true', default=False)
    parser.add_argument('--frozen_snn', action='store_true', default=False)
    parser.add_argument('--frozen_lbm', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)
    args = parser.parse_args()

    datasets = ['brennan2019', 'broderick2019', 'ISRUC', 'FACED', 'SEED-V', 'BCIC2020',
                'TUEV', 'PhysioNet-MI', 'SHU-MI', 'TUAB', 'Mumtaz2016', 'MentalArithmetic']

    plot_datasets = ['broderick2019', 'PhysioNet-MI', 'BCIC2020',
                     'TUEV', 'SEED-V', 'TUAB']

    for dataset in datasets:
        if dataset not in plot_datasets:
            continue

        args.datasets = dataset
        ckpt_snn = rf"E:\NIPS2026\ckpt\cross-subject-task-snn\{dataset}.pth"

        if dataset == 'ISRUC':
            args.n_classes = 5
            args.n_subjects = 100
            args.n_channels = 6
            args.sr = 200
            args.fps = 1
            data_loaders = data_isruc.LoadDataset(args)
            data_loaders = data_loaders.get_data_loader()
            snn_model = SAS(args)
        elif dataset == 'FACED':
            args.n_classes = 9
            args.n_channels = 32
            args.sr = 200
            args.fps = 1
            data_loaders = data_faced.LoadDataset(args)
            data_loaders = data_loaders.get_data_loader()
            snn_model = SAS(args)
        elif dataset == 'PhysioNet-MI':
            args.n_classes = 4
            args.n_channels = 64
            args.sr = 200
            args.fps = 5
            data_loaders = data_physio.LoadDataset(args)
            data_loaders = data_loaders.get_data_loader()
            snn_model = SAS(args)
        elif dataset == 'Mumtaz2016':
            args.n_subjects = 64
            args.n_channels = 19
            args.sr = 200
            args.fps = 10
            data_loaders = data_mumtaz2016.LoadDataset(args)
            data_loaders = data_loaders.get_data_loader()
            snn_model = SAS(args)
        elif dataset == 'MentalArithmetic':
            args.n_subjects = 36
            args.n_channels = 20
            args.sr = 200
            args.fps = 10
            data_loaders = data_mental.LoadDataset(args)
            data_loaders = data_loaders.get_data_loader()
            snn_model = SAS(args)
        elif dataset == 'SHU-MI':
            args.n_channels = 32
            args.sr = 200
            args.fps = 5
            data_loaders = data_shumi.LoadDataset(args)
            data_loaders = data_loaders.get_data_loader()
            snn_model = SAS(args)
        elif dataset == 'SEED-VIG':
            args.n_subjects = 1
            args.n_channels = 17
            args.sr = 200
            args.fps = 3
            data_loaders = data_seedvig.LoadDataset(args)
            data_loaders = data_loaders.get_data_loader()
            snn_model = SAS(args)
        elif dataset == 'SEED-V':
            args.n_classes = 5
            args.n_subjects = 16
            args.n_channels = 62
            args.sr = 200
            args.fps = 10
            data_loaders = data_seedv.LoadDataset(args)
            data_loaders = data_loaders.get_data_loader()
            snn_model = SAS(args)
        elif dataset == 'TUAB':
            args.n_subjects = 1
            args.n_channels = 16
            args.sr = 200
            args.fps = 2
            data_loaders = data_tuab.LoadDataset(args)
            data_loaders = data_loaders.get_data_loader()
            snn_model = SAS(args)
        elif dataset == 'TUEV':
            args.n_classes = 6
            args.n_subjects = 1
            args.n_channels = 16
            args.sr = 200
            args.fps = 4
            data_loaders = data_tuev.LoadDataset(args)
            data_loaders = data_loaders.get_data_loader()
            snn_model = SAS(args)
        elif dataset == 'BCIC2020':
            args.n_classes = 5
            args.n_subjects = 15
            args.n_channels = 16
            args.sr = 200
            args.fps = 10
            data_loaders = data_bcic2020.LoadDataset(args)
            data_loaders = data_loaders.get_data_loader()
            snn_model = SAS(args)
        elif dataset == 'broderick2019':
            args.n_negatives = 100
            args.n_subjects = 19
            args.n_channels = 128
            args.sr = 120
            args.fps = 5
            data_loaders = data_broderick2019.LoadDataset(args)
            data_loaders = data_loaders.get_data_loader()
            snn_model = SAS(args)
        elif dataset == 'brennan2019':
            args.n_negatives = 200
            args.n_subjects = 32
            args.n_channels = 60
            args.sr = 120
            args.fps = 3
            data_loaders = data_brennan2019.LoadDataset(args)
            data_loaders = data_loaders.get_data_loader()
            snn_model = SAS(args)

        snn_state = torch.load(ckpt_snn, map_location='cuda', weights_only=False)
        snn_model.load_state_dict(snn_state, strict=False)
        snn_model = snn_model.to('cuda')

        len_devs = []
        lens = []
        subjectss = []
        for x, y, events, subjects in data_loaders['test']:
            B, L, C, T = x.shape
            n_frames = round(T / args.sr * args.fps)
            events = rearrange(events, 'B L t P C -> (B L) t P C', B=B, L=L).to('cuda')
            spike_idxes = snn_model(events)
            ss = torch.tensor([s[0].item() if len(s) != 0 else n_frames for s in spike_idxes])
            len_dev = ss
            len_devs.extend(len_dev.tolist())

            len_ss = ss[1:] - ss[:-1] + n_frames
            lens.extend(len_ss.tolist())
            subjectss.extend(subjects.flatten().tolist())

        len_devs = torch.tensor(len_devs).float()
        lens = torch.tensor(lens).float()
        subjectss = torch.tensor(subjectss).int()

        print(f"{dataset}:")
        print(f"n_frames: {n_frames}")
        print(f"dev of slice points: {torch.mean(len_devs):.3f}, {torch.std(len_devs)}")
        print(f"dev of slice points in seconds: {torch.mean(len_devs) / args.fps:.3f}, {torch.std(len_devs) / args.fps}")
        print(f"length: {torch.mean(lens):.3f}, {torch.std(lens)}")
        print(f"length in seconds: {torch.mean(lens) / args.fps:.3f}, {torch.std(lens) / args.fps}")
        print("\n\n")
        torch.cuda.empty_cache()

        if dataset in plot_datasets:
            subjectss = subjectss[len_devs != n_frames]
            len_devs = len_devs[len_devs != n_frames]
            unique_subjects = torch.unique(subjectss)

            num_subjects = 6
            colors = ['mediumpurple', 'steelblue', 'lightseagreen', 'olivedrab', 'goldenrod', 'darkred']
            labels = [f'Subject {i}' for i in range(num_subjects)]
            bins = np.arange(0, n_frames + 1) - 0.5  # bin edges for integer data

            # Create subplot
            fig, axes = plt.subplots(num_subjects, 1, figsize=(8, 1.5 * num_subjects), sharex=True)

            selected_subjects = subjectss[torch.randperm(len(subjectss))[:num_subjects]]
            for i, subj in enumerate(selected_subjects):
                ax = axes[i]
                data = len_devs[subjectss == subj].numpy()
                ax.hist(data, bins=bins, color=colors[i], alpha=0.4)

                # ax.axvline(0, color='black', linestyle='--')
                # ax.axvline(n_frames - 1, color='black', linestyle='--')
                ax.set_xlim([-1, n_frames])
                ax.set_ylabel(labels[i], rotation=0, labelpad=40, va='center')
                ax.set_yticks([])
                ax.set_xticks([])

            for ax in axes:
                for spine in ax.spines.values():
                    spine.set_visible(False)

            plt.tight_layout()
            plt.show()
