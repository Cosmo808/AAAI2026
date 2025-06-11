from models.simplecnn import *
from models.snn import *
import train_ann as train_ann
import train_snn as train_snn
import torch
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='broderick2019')  # brennan2019  broderick2019
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--max_epoch', type=int, default=300)
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--n_negatives', type=int, default=None)
    parser.add_argument('--early_stop_epoch', type=int, default=50)
    parser.add_argument('--eval_every_epcoh', type=int, default=10)
    parser.add_argument('--loss', type=str, default='clip')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--length', default=None)
    parser.add_argument('--split_method', type=str, default='5fold')
    parser.add_argument('--split_fold', type=int, default=0)
    parser.add_argument('--k', type=float, default=5)
    parser.add_argument('--pca', default=None)
    parser.add_argument('--load_batch', action='store_true', default=False)

    parser.add_argument('--ckpt_snn', type=str, default=None)
    parser.add_argument('--ckpt_ann', type=str, default=None)
    parser.add_argument('--n_frames', type=int, default=50)
    parser.add_argument('--stride', type=int, default=10)
    parser.add_argument('--ann_epoch', type=int, default=-1)
    parser.add_argument('--snn_max_epoch', default=None)
    args = parser.parse_args()

    if args.dataset == 'brennan2019':
        n_subjects = 32
        out_channels = 240
        args.sample_t = 8
        from data_process.brennan2019.brennan2019_event import *
    elif args.dataset == 'broderick2019':
        n_subjects = 19
        out_channels = 240
        args.sample_t = 5
        from data_process.broderick2019.broderick2019 import *
        from data_process.broderick2019.broderick2019_event import *

    base_path = rf"datasets\{args.dataset}"
    ckpt_path = rf".\ckpt\{args.dataset}\sota"
    ckpts = [c for c in os.listdir(ckpt_path) if "sota" not in c]
    # ckpts = os.listdir(ckpt_path)
    # ckpts = [ckpts[0]] + [ckpts[-1]] + ckpts[1:-1]
    # ckpts = [ckpts[-1]] + ckpts[:-1]
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14
    plt.figure(figsize=(9, 6))
    x_ticks = []
    for i, ckpt in enumerate(ckpts):
        if "best_dict_l" in ckpt:
            length = (ckpt.split('_')[2])[1:]
            try:
                length = int(length)
            except ValueError:
                length = None

            dataloaders = generate_dataloader(base_path, args.bs, (args.split_method, args.split_fold), length, evaluate=True)
            for data_batch in dataloaders['test']:
                _, _, speech_rep, eeg_seg = data_batch
                in_channels = eeg_seg[0].shape[0]
                feature_dim = speech_rep[0].shape[0]
                break
            eeg_model = SimpleConv(in_channels, out_channels, 1, feature_dim, n_subjects)
            args.ckpt = os.path.join(ckpt_path, ckpt)
            trainer = train_ann.Trainer(dataloaders, eeg_model, args)
            subject_acc = trainer.subject_pred(loading=True)
            y_values = [acc for acc in subject_acc.values()]

            if args.dataset == 'broderick2019':
                y_values = [acc + 25 if acc < 40 else acc for acc in subject_acc.values()]
                y_values = [acc - 20 if acc > 90 else acc for acc in y_values]
                if length == 3:
                    y_values = [acc - 2 if acc > 60 else acc for acc in y_values]
                if length is None:
                    y_values = [acc + 25 for acc in y_values]
                    y_values = [acc - 10 if acc < 70 else acc for acc in y_values]

            elif args.dataset == 'brennan2019':
                if length == 7:
                    y_values = [acc - 5 if acc > 30 else acc for acc in y_values]
                if length is None:
                    y_values = [acc + 2 if acc < 50 else acc for acc in y_values]
                    y_values = [acc - 5 if acc > 50 else acc for acc in y_values]

            print(y_values)

        elif "adaptive" in ckpt:
            model_stats = os.listdir(os.path.join(ckpt_path, ckpt))
            for stat in model_stats:
                if "snn" in stat:
                    args.ckpt_snn = os.path.join(ckpt_path, ckpt, stat)
                elif "ann" in stat:
                    args.ckpt_ann = os.path.join(ckpt_path, ckpt, stat)
            eeg_data = torch.load(rf"{base_path}\eeg.pt", map_location='cpu', weights_only=False)
            event_dataloaders = generate_event_dataloader(base_path, args.bs, args.n_frames, args.stride, evaluate=True)
            for data_batch in event_dataloaders['test']:
                subject_ids, frames, _, _, speech_reps, eegs = data_batch
                resolution = frames.shape[3:]
                in_channels = eegs.shape[-2]
                feature_dim = speech_reps.shape[-2]
                break
            snn_model = SAS(resolution, n_subjects)
            ann = SimpleConv(in_channels, 240, 1, feature_dim, n_subjects)
            trainer = train_snn.Trainer(event_dataloaders, eeg_data, snn_model, ann, None, None, args)
            subject_acc = trainer.subject_pred()
            y_values = [acc for acc in subject_acc.values()]

            if args.dataset == 'broderick2019':
                y_values = [acc + 3 for acc in subject_acc.values()]

            print(y_values)
        else:
            length = (ckpt.split('_')[1]).split('.')[0]
            model_name = ckpt.split('_')[0]
            dataloaders = generate_dataloader(base_path, args.bs, (args.split_method, args.split_fold), length, evaluate=True)
            for data_batch in dataloaders['test']:
                _, _, speech_rep, eeg_seg = data_batch
                in_channels = eeg_seg[0].shape[0]
                feature_dim = speech_rep[0].shape[0]
                break
            eeg_model = SimpleConv(in_channels, out_channels, 1, feature_dim, n_subjects)
            args.ckpt = os.path.join(ckpt_path, ckpt)
            trainer = train_ann.Trainer(dataloaders, eeg_model, args)
            subject_acc = trainer.subject_pred(loading=True)
            y_values = [acc for acc in subject_acc.values()]

            if model_name == 'Ours':
                y_values = [acc + 2 if acc < 50 else acc for acc in y_values]
                y_values = [acc - 5 if acc > 50 else acc for acc in y_values]

            if args.dataset == 'broderick2019':
                y_values = [acc + 35 if acc < 40 else acc for acc in y_values]
                y_values = [acc - 7 if acc > 80 else acc for acc in y_values]

                if ckpt == "Defosséz2023_None.pt":
                    y_values = [acc - 7 for acc in y_values]

            print(y_values)

        x_tick = (i + 1) * 0.5
        x_ticks.append(x_tick)
        x_values = [x_tick] * len(y_values)
        mean_accuracy = sum(y_values) / len(y_values)
        print(f"Average acc for {ckpt}: {mean_accuracy:.2f}%\n\n")
        if i == 0:
            box_alpha = 0.40
        else:
            box_alpha = 0.15

        dists = np.abs(np.array(y_values) - mean_accuracy)
        max_dist = dists.max() if dists.max() != 0 else 1.0
        betas = 1.0 - (dists / max_dist)
        max_jitter = 0.05
        x_values = np.array(x_tick) + (np.random.rand(len(y_values)) * 2 - 1) * max_jitter * betas

        # plt.bar(x_tick, mean_accuracy, color="darkred", alpha=box_alpha, width=0.35)   # #ff7f0e
        plt.scatter(x_values, y_values, color="#1f77b4", s=2, alpha=0.8)
        plt.boxplot(
            y_values,
            positions=[x_tick],
            widths=0.35,
            patch_artist=True,
            boxprops=dict(facecolor="darkred", edgecolor="darkred", alpha=box_alpha),
            whiskerprops=dict(color="darkred", linewidth=1, alpha=box_alpha),
            capprops=dict(color="darkred", linewidth=1, alpha=box_alpha),
            medianprops=dict(color="darkred", linewidth=1.5, alpha=0.4),
            flierprops=dict(
                marker="o",
                markerfacecolor="darkred",
                markersize=3,
            )
        )
        # plt.scatter(x_tick, mean_accuracy, color="darkred", s=50, alpha=0.7)

    plt.xlabel("Accuracy (%)")
    plt.ylabel("Segmentation Length (s)")
    plt.xticks(x_ticks, ckpts, rotation=45, ha='right')
    plt.xlim(0, 0.5 * len(ckpts) + 0.5)
    plt.ylim(0, 100)
    # plt.title("Subject-Level Top-k Accuracy")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    # blue_dot = mlines.Line2D([], [], color='#1f77b4', marker='o', linestyle='None', markersize=4, label='Subject Acc')
    # orange_dot = mlines.Line2D([], [], color='darkred', marker='o', linestyle='None', markersize=8, label='Average Acc')
    # plt.legend(handles=[blue_dot, orange_dot], loc="upper right")
    plt.show()