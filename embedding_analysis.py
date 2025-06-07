import os
import argparse
import pdb

import spacy
import mne
import pandas as pd
import numpy as np
from laser_encoders import LaserEncoderPipeline
from wordfreq import zipf_frequency as zipf
from tqdm import tqdm
import torch
import torch.nn.functional as F

from data_loader import data_isruc, data_broderick2019, data_brennan2019
from models import model_isruc, model_broderick2019, model_brennan2019
from trainers import trainer_isruc, trainer_broderick2019, trainer_brennan2019
from models.snn import SAS
from models.utils import Brain2Event, wav_processor

from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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

    parser.add_argument('--datasets', type=str, default='broderick2019')  # brennan2019  broderick2019  ISRUC
    parser.add_argument('--model', type=str, default='cbramod')  # simplecnn  cbramod  labram
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
    args = parser.parse_args()

    if args.datasets == 'broderick2019':
        args.n_subjects = 19
        args.n_channels = 128
        args.n_slice = 1
        args.sr = 120
        args.fps = 5
        sample_t = 5
        data_loaders = data_broderick2019.LoadDataset(args)
        data_loader = data_loaders.get_alldata()
        eeg_model = model_broderick2019.Model(args)
        snn_model = SAS(args)
        trainer = trainer_broderick2019

    # prepare
    args.ckpt_ann = rf"E:\NIPS2026\ckpt\{args.datasets}\{args.model}\ann_epoch44_10@50_0.76576_10@All_0.22203.pth"
    # args.ckpt_snn = rf"E:\NIPS2026\ckpt\{args.datasets}\{args.model}+sas-brain\snn_epoch88_spike_0.00148.pth"
    word_encoder = spacy.load("en_core_web_lg")
    sentence_encoder = LaserEncoderPipeline(lang="eng_Latn")

    trainer = trainer.Trainer({'train': data_loader}, eeg_model, snn_model, None, None, args)
    scoring = make_scorer(lambda yt, yp: pearsonr(yt, yp)[0])
    model = make_pipeline(StandardScaler(), Ridge())
    cv = KFold(5, shuffle=True)

    # evaluate
    if os.path.isfile(rf"{args.base_dir}\cache\preds.pth"):
        preds = torch.load(rf"{args.base_dir}\cache\preds.pth", map_location='cpu', weights_only=False)
        wordfs = torch.load(rf"{args.base_dir}\cache\wordfs.pth", map_location='cpu', weights_only=False)
        word_embs = torch.load(rf"{args.base_dir}\cache\word_embs.pth", map_location='cpu', weights_only=False)
        sentence_embs = torch.load(rf"{args.base_dir}\cache\sentence_embs.pth", map_location='cpu', weights_only=False)
        scores = torch.load(rf"{args.base_dir}\cache\scores.pth", map_location='cpu', weights_only=False)
        print(f"Saved scores: {torch.mean(scores, dim=0)}/{torch.std(scores, dim=0)} (mean/std)")
    else:
        preds = []
        wordfs = []
        word_embs = []
        sentence_embs = []

        for x, y, events, subjects, texts in tqdm(data_loader):
            with torch.no_grad():
                # x_sas, y_sas = trainer.snn_one_batch(x, y, events, subjects, slice=True)
                pred = trainer.ann(x.to(trainer.device), subjects.to(trainer.device)).detach().cpu()
                preds.append(pred)

                text = [item for t in texts for item in t]
                text = [t.replace('"', '').replace('\n', '').replace('\r', '') for t in text]

                for t in text:
                    word = t.split(' ')

                    wordf = torch.tensor([zipf(w, 'en') for w in word]).view(1, 1, -1)
                    wordf = F.interpolate(wordf, size=pred.shape[1], mode='linear', align_corners=False).view(-1)
                    word_emb = torch.stack([torch.tensor(we.vector) for we in word_encoder(t)]).flatten().view(1, 1, -1)
                    word_emb = F.interpolate(word_emb, size=pred.shape[1], mode='linear', align_corners=False).view(-1)
                    sentence_emb = torch.tensor(sentence_encoder.encode_sentences([t])).view(1, 1, -1)
                    sentence_emb = F.interpolate(sentence_emb, size=pred.shape[1], mode='linear', align_corners=False).view(-1)

                    wordfs.append(wordf)
                    word_embs.append(word_emb)
                    sentence_embs.append(sentence_emb)

        preds = torch.cat(preds, dim=0)
        wordfs = torch.stack(wordfs)
        word_embs = torch.stack(word_embs)
        sentence_embs = torch.stack(sentence_embs)
        assert preds.shape[0] == wordfs.shape[0]
        cache_dir = rf"{args.base_dir}\cache"
        torch.save(preds, rf"{cache_dir}\preds.pth")
        torch.save(wordfs, rf"{cache_dir}\wordfs.pth")
        torch.save(word_embs, rf"{cache_dir}\word_embs.pth")
        torch.save(sentence_embs, rf"{cache_dir}\sentence_embs.pth")

    scores = torch.zeros([pred.shape[1], 3])
    for i in range(pred.shape[1]):
        if i % 10 == 0:
            print(f"i/{pred.shape[1]}")
        pred = preds[:, i, :]
        wordf = wordfs[:, i]
        word_emb = word_embs[:, i]
        sentence_emb = sentence_embs[:, i]

        score_wf = cross_val_score(model, pred, wordf, scoring=scoring, cv=cv, n_jobs=-1)
        scores[i, 0] = np.nanmean(score_wf)
        score_we = cross_val_score(model, pred, word_emb, scoring=scoring, cv=cv, n_jobs=-1)
        scores[i, 1] = np.nanmean(score_we)
        score_se = cross_val_score(model, pred, sentence_emb, scoring=scoring, cv=cv, n_jobs=-1)
        scores[i, 2] = np.nanmean(score_se)

    score_mean = torch.mean(scores, dim=0)
    score_std = torch.std(scores, dim=0)
    torch.save(scores, rf"{args.base_dir}\cache\scores.pth")
    print(score_mean, score_std)
