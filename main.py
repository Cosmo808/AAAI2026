import argparse
import random

from models.snn import SAS
from models.utils import *

from data_loader import data_isruc, data_broderick2019, data_brennan2019, data_mumtaz2016, data_mental, \
    data_tuab, data_tuev, data_bcic2020, data_schoffelen2019, data_gwilliams2022
from models import model_isruc, model_broderick2019, model_brennan2019, model_mumtaz2016, model_mental, \
    model_tuab, model_tuev, model_bcic2020, model_schoffelen2019, model_gwilliams2022
from trainers import trainer_isruc, trainer_broderick2019, trainer_brennan2019, trainer_mumtaz2016, trainer_mental, \
    trainer_tuab, trainer_tuev, trainer_bcic2020, trainer_schoffelen2019, trainer_gwilliams2022


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
                        choices=['brennan2019', 'broderick2019', 'schoffelen2019', 'gwilliams2020',
                                 'ISRUC', 'TUEV', 'BCIC2020',
                                 'Mumtaz2016', 'MentalArithmetic', 'TUAB'])
    parser.add_argument('--model', type=str, default='cbramod', choices=['simplecnn', 'cbramod', 'labram'])
    parser.add_argument('--n_negatives', type=int, default=None)
    parser.add_argument('--n_subjects', type=int, default=None)
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

    # check if evaluate
    if args.eval:
        args.frozen_ann = True
        args.frozen_snn = True
        args.max_epoch = 0

    # clear cache
    # for file_path in glob.glob(os.path.join(rf"{args.base_dir}\cache", "*")):
    #     if os.path.isfile(file_path):
    #         os.remove(file_path)

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
        args.sr = 200
        args.fps = 1
        data_loaders = data_isruc.LoadDataset(args)
        data_loaders = data_loaders.get_data_loader()
        eeg_model = model_isruc.Model(args)
        snn_model = SAS(args)
        trainer = trainer_isruc
    elif args.datasets == 'Mumtaz2016':
        args.n_classes = 2
        args.n_subjects = 64
        args.n_channels = 19
        args.sr = 200
        args.fps = 10
        data_loaders = data_mumtaz2016.LoadDataset(args)
        data_loaders = data_loaders.get_data_loader()
        eeg_model = model_mumtaz2016.Model(args)
        snn_model = SAS(args)
        trainer = trainer_mumtaz2016
    elif args.datasets == 'MentalArithmetic':
        args.n_classes = 2
        args.n_subjects = 36
        args.n_channels = 20
        args.sr = 200
        args.fps = 10
        data_loaders = data_mental.LoadDataset(args)
        data_loaders = data_loaders.get_data_loader()
        eeg_model = model_mental.Model(args)
        snn_model = SAS(args)
        trainer = trainer_mental
    elif args.datasets == 'TUAB':
        args.n_classes = 2
        args.n_subjects = 1
        args.n_channels = 16
        args.sr = 200
        args.fps = 2
        data_loaders = data_tuab.LoadDataset(args)
        data_loaders = data_loaders.get_data_loader()
        eeg_model = model_tuab.Model(args)
        snn_model = SAS(args)
        trainer = trainer_tuab
    elif args.datasets == 'TUEV':
        args.n_classes = 6
        args.n_subjects = 1
        args.n_channels = 16
        args.sr = 200
        args.fps = 4
        data_loaders = data_tuev.LoadDataset(args)
        data_loaders = data_loaders.get_data_loader()
        eeg_model = model_tuev.Model(args)
        snn_model = SAS(args)
        trainer = trainer_tuev
    elif args.datasets == 'BCIC2020':
        args.n_classes = 5
        args.n_subjects = 15
        args.n_channels = 16
        args.sr = 200
        args.fps = 10
        data_loaders = data_bcic2020.LoadDataset(args)
        data_loaders = data_loaders.get_data_loader()
        eeg_model = model_bcic2020.Model(args)
        snn_model = SAS(args)
        trainer = trainer_bcic2020
    elif args.datasets == 'broderick2019':
        args.n_negatives = 100
        args.n_subjects = 19
        args.n_channels = 128
        args.sr = 120
        args.fps = 5
        data_loaders = data_broderick2019.LoadDataset(args)
        data_loaders = data_loaders.get_data_loader()
        eeg_model = model_broderick2019.Model(args)
        snn_model = SAS(args)
        trainer = trainer_broderick2019
    elif args.datasets == 'brennan2019':
        args.n_negatives = 200
        args.n_subjects = 32
        args.n_channels = 60
        args.sr = 120
        args.fps = 3
        data_loaders = data_brennan2019.LoadDataset(args)
        data_loaders = data_loaders.get_data_loader()
        eeg_model = model_brennan2019.Model(args)
        snn_model = SAS(args)
        trainer = trainer_brennan2019
    elif args.datasets == 'schoffelen2019':
        args.n_negatives = 100
        args.n_subjects = 30
        args.n_channels = 273
        args.sr = 120
        args.fps = 4
        data_loaders = data_schoffelen2019.LoadDataset(args)
        data_loaders = data_loaders.get_data_loader()
        eeg_model = model_schoffelen2019.Model(args)
        snn_model = SAS(args)
        trainer = trainer_schoffelen2019
    elif args.datasets == 'gwilliams2020':
        args.n_negatives = 100
        args.n_subjects = 27
        args.n_channels = 208
        args.sr = 120
        args.fps = 2
        data_loaders = data_gwilliams2022.LoadDataset(args)
        data_loaders = data_loaders.get_data_loader()
        eeg_model = model_gwilliams2022.Model(args)
        snn_model = SAS(args)
        trainer = trainer_gwilliams2022

    # optimizer and scheduler
    backbone_params = []
    other_params = []
    for name, param in eeg_model.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    for frozen, params in [
        (args.frozen_ann, eeg_model.parameters()),
        (args.frozen_snn, snn_model.parameters()),
        (args.frozen_lbm, backbone_params),
    ]:
        if frozen:
            for p in params: p.requires_grad = False

    if args.multi_lr:
        optimizer = torch.optim.AdamW([
            {'params': other_params, 'lr': args.lr, 'name': 'ann'},
            {'params': snn_model.parameters(), 'lr': args.lr, 'name': 'snn'},
            {'params': backbone_params, 'lr': args.lr / 5, 'name': 'lbm'}
        ], betas=(0.9, 0.999), weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW([
            {'params': eeg_model.parameters(), 'lr': args.lr, 'name': 'ann'},
            {'params': snn_model.parameters(), 'lr': args.lr, 'name': 'snn'},
        ], betas=(0.9, 0.999), weight_decay=args.weight_decay)

    scheduler_ann = GroupCosineAnnealingLR(optimizer, group_index=0, T_max=args.max_epoch * len(data_loaders['train']), eta_min=1e-6, verbose=False, name='ann')
    scheduler_snn = GroupCosineAnnealingLR(optimizer, group_index=1, T_max=args.max_epoch * len(data_loaders['train']), eta_min=1e-6, verbose=False, name='snn')
    schedulers = [scheduler_ann, scheduler_snn]

    print(f"The ann contains {sum(p.numel() for p in other_params)} parameters.")
    print(f"The snn contains {sum(p.numel() for p in snn_model.parameters())} parameters.")
    print(f"The backbone contains {sum(p.numel() for p in backbone_params)} parameters.")
    
    # train
    trainer = trainer.Trainer(data_loaders, eeg_model, snn_model, optimizer, schedulers, args)
    trainer.train()

