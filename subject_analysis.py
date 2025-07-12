import argparse
import pdb
import random
import numpy as np
import torch
import os
import glob
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score

from models.snn import SAS
from models.utils import *

from data_loader import data_isruc, data_broderick2019, data_brennan2019, data_mumtaz2016, data_mental, data_shumi,\
    data_tuab, data_tuev, data_bcic2020, data_schoffelen2019, data_gwilliams2022, data_seedvig, data_seedv, data_faced, data_physio
from models import model_isruc, model_broderick2019, model_brennan2019, model_mumtaz2016, model_mental, model_shumi,\
    model_tuab, model_tuev, model_bcic2020, model_schoffelen2019, model_gwilliams2022, model_seedvig, model_seedv, model_faced, model_physio
from trainers import trainer_isruc, trainer_broderick2019, trainer_brennan2019, trainer_mumtaz2016, trainer_mental, trainer_shumi,\
    trainer_tuab, trainer_tuev, trainer_bcic2020, trainer_schoffelen2019, trainer_gwilliams2022, trainer_seedvig, trainer_seedv, trainer_faced, trainer_physio


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default=r"E:\NIPS2026")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--early_stop_epoch', type=int, default=20)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--multi_lr', action='store_true', default=False)
    parser.add_argument('--weight_decay', type=float, default=5e-2)
    parser.add_argument('--grad_clip', type=float, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    parser.add_argument('--datasets', type=str, default='MentalArithmetic')
    parser.add_argument('--model', type=str, default='cbramod')  # simplecnn  cbramod  labram
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
    args = parser.parse_args()

    # setup seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # obtain dataloader, model, trainer
    if args.datasets == 'ISRUC':
        args.n_classes = 5
        args.n_subjects = 100
        args.n_channels = 6
        args.n_slice = 1
        args.sr = 200
        args.fps = 1
        data_loaders = data_isruc.LoadDataset(args)
        data_loaders = data_loaders.get_data_loader()
        eeg_model = model_isruc.Model(args)
        snn_model = SAS(args)
        trainer = trainer_isruc
        ckpts = [
            r'cbramod\ann_epoch2_acc_0.78914_kappa_0.74099_f1_0.79926.pth',
            r'cbramod\ann_acc_0.78219_kappa_0.74172_f1_0.79994.pth',
            (r'cbramod+sas-brain\ann_best_acc_0.79315_kappa_0.75438_f1_0.81432.pth', r'cbramod+sas-brain\snn_best_spike_0.01052.pth')
        ]
    elif args.datasets == 'PhysioNet-MI':
        args.n_classes = 4
        args.n_channels = 64
        args.sr = 200
        args.fps = 5
        data_loaders = data_physio.LoadDataset(args)
        data_loaders = data_loaders.get_data_loader()
        eeg_model = model_physio.Model(args)
        snn_model = SAS(args)
        trainer = trainer_physio
        ckpts = [
            r'cbramod\ann_epoch50_acc_0.63331_kappa_0.51164_f1_0.63447.pth',
            r'cbramod\ann_epoch0_acc_0.62014_kappa_0.49411_f1_0.62054.pth',
            r'cbramod\ann_epoch7_acc_0.63933_kappa_0.51938_f1_0.64210.pth',
            (r'cbramod+sas-brain\ann_epoch40_acc_0.64135_kappa_0.52234_f1_0.64341.pth', r'cbramod+sas-brain\snn_epoch40_spike_0.00486.pth')
        ]
    elif args.datasets == 'SEED-V':
        args.n_classes = 5
        args.n_subjects = 16
        args.n_channels = 62
        args.sr = 200
        args.fps = 10
        data_loaders = data_seedv.LoadDataset(args)
        data_loaders = data_loaders.get_data_loader()
        eeg_model = model_seedv.Model(args)
        snn_model = SAS(args)
        trainer = trainer_seedv
        ckpts = [
            r'cbramod\ann_epoch6_acc_0.40055_kappa_0.25656_f1_0.41210.pth',
            r'cbramod\ann_epoch7_acc_0.40118_kappa_0.25804_f1_0.41251.pth',
            r'cbramod\ann_epoch0_acc_0.39917_kappa_0.25488_f1_0.40776.pth',
            (r'cbramod+sas-brain\ann_epoch9_acc_0.40538_kappa_0.26299_f1_0.41667.pth', r'cbramod+sas-brain\snn_epoch9_spike_0.09794.pth')
        ]
    elif args.datasets == 'broderick2019':
        args.n_subjects = 19
        args.n_channels = 128
        args.n_slice = 1
        args.sr = 120
        args.fps = 5
        data_loaders = data_broderick2019.LoadDataset(args)
        data_loaders = data_loaders.get_data_loader()
        eeg_model = model_broderick2019.Model(args)
        snn_model = SAS(args)
        trainer = trainer_broderick2019
        ckpts = [
            r'cbramod\ann_epoch23_10@50_0.77831_10@All_0.22712.pth',
            r'cbramod\ann_epoch50_10@50_0.75661_10@All_0.21322.pth',
            r'cbramod\ann_epoch61_10@50_0.75797_10@All_0.22542.pth',
            r'cbramod\ann_epoch38_10@50_0.75831_10@All_0.20678.pth',
            r'cbramod\ann_epoch38_10@50_0.72305_10@All_0.18339.pth',
            (r'cbramod+sas-brain\ann_epoch0_10@50_0.81898_10@All_0.23356.pth', r'cbramod+sas-brain\snn_epoch0_spike_0.11116.pth')
        ]
    elif args.datasets == 'brennan2019':
        args.n_subjects = 32
        args.n_channels = 60
        args.n_slice = 1
        args.sr = 120
        args.fps = 3
        data_loaders = data_brennan2019.LoadDataset(args)
        data_loaders = data_loaders.get_data_loader()
        eeg_model = model_brennan2019.Model(args)
        snn_model = SAS(args)
        trainer = trainer_brennan2019
        ckpts = [
            r'cbramod\ann_epoch49_10@50_0.81750_10@All_0.16100.pth',
            r'cbramod\ann_epoch0_10@50_0.85000_10@All_0.17100.pth',
            r'cbramod\ann_epoch46_10@50_0.81450_10@All_0.15050.pth',
            r'cbramod\ann_epoch49_10@50_0.76700_10@All_0.11600.pth',
            r'cbramod\ann_epoch96_10@50_0.86200_10@All_0.22850.pth',
            (r'cbramod+sas-brain\ann_epoch93_10@50_0.90300_10@All_0.29850.pth', r'cbramod+sas-brain\snn_epoch93_spike_0.01630.pth')
        ]

    for i, ckpt in enumerate(ckpts):
        if len(ckpt) == 2:
            args.ckpt_ann = rf"{args.base_dir}\ckpt\{args.datasets}\{ckpt[0]}"
            args.ckpt_snn = rf"{args.base_dir}\ckpt\{args.datasets}\{ckpt[1]}"
        else:
            args.ckpt_ann = rf"{args.base_dir}\ckpt\{args.datasets}\{ckpt}"

        trainer_i = trainer.Trainer(data_loaders, eeg_model, snn_model, None, None, args)
        trainer_i.ann.eval()

        subject_metric = defaultdict(list)
        subject_metric_1 = defaultdict(list)
        for x, y, events, subjects in tqdm(data_loaders['test']):
            trainer_i.iter += 1

            if args.datasets in ['broderick2019', 'brennan2019']:
                if args.ckpt_snn is not None:
                    _ = trainer_i.snn_one_batch(x, y, events, subjects, training=False)
                correct_all, correct_50 = trainer_i.ann_one_batch(x, y, events, subjects, training=False)
                metrics = correct_50
            else:
                if args.ckpt_snn is not None:
                    _ = trainer_i.snn_one_batch(x, y, events, training=False)
                outputs = trainer_i.ann_one_batch(x, y, events, training=False)
                metrics = torch.tensor([outputs[0], outputs[1]]).t()

            subjects = subjects.flatten()
            for s, metric in zip(subjects, metrics):
                subject_metric[s.item()].append(metric)

        try:
            for subj in subject_metric:
                m = torch.stack(subject_metric[subj])
                truth, pred = m[:, 0], m[:, 1]
                subject_metric[subj] = balanced_accuracy_score(truth, pred)
            subject_metric = [subject_metric[subj] for subj in subject_metric]
        except:
            subject_metric = [sum(subject_metric[subj]) / len(subject_metric[subj]) for subj in subject_metric]

        print(subject_metric)

