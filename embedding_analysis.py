import gc
import os
import argparse
import pdb
import spacy
import mne
import pandas as pd
import numpy as np
from laser_encoders import LaserEncoderPipeline, initialize_tokenizer
from wordfreq import zipf_frequency as zipf
from tqdm import tqdm
import torch
import torch.nn.functional as F
from scipy.stats import t as student_t
from collections import defaultdict

from data_loader import data_isruc, data_broderick2019, data_brennan2019
from models import model_isruc, model_broderick2019, model_brennan2019
from trainers import trainer_isruc, trainer_broderick2019, trainer_brennan2019
from models.snn import SAS
from models.utils import RidgeRegression, ConvLinear

from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def evaluate_model(model_cls, X, y, epochs=500, cv=2, lr=1e-3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx].to(device), X[test_idx]
        y_train, y_test = y[train_idx].to(device), y[test_idx]

        if issubclass(model_cls, ConvLinear):
            model = model_cls(dim1=X.shape[-2], dim2=X.shape[-1])
        elif issubclass(model_cls, RidgeRegression):
            model = model_cls(input_dim=X.shape[-1])
        model.train_model(X_train, y_train.unsqueeze(dim=-1), epochs=epochs, lr=lr, verbose=False)
        del X_train, y_train
        gc.collect()
        torch.cuda.empty_cache()

        y_pred = model.predict(X_test.to(device)).view(-1)
        corr, pval = pearson_corr_torch(y_pred, y_test.to(device))
        # print(corr, pval)
        if np.isnan(corr) or pval > 0.1:
            scores.append(float('nan'))
        else:
            scores.append(corr)
    return np.abs(np.nanmean(scores)), np.abs(np.nanstd(scores))


def pearson_corr_torch(x: torch.Tensor, y: torch.Tensor):
    x = x.float()
    y = y.float()
    n = x.numel()

    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    x_centered = x - x_mean
    y_centered = y - y_mean

    numerator = torch.sum(x_centered * y_centered)
    denominator = torch.sqrt(torch.sum(x_centered ** 2)) * torch.sqrt(torch.sum(y_centered ** 2))
    r = numerator / denominator
    r = torch.clamp(r, -0.999999, 0.999999)
    t = r * torch.sqrt(torch.tensor(n - 2, dtype=torch.float32)) / torch.sqrt(1 - r ** 2)
    p = 2 * (1 - student_t.cdf(torch.abs(t).item(), df=n - 2))
    return r.item(), p


def process_sentences_with_spikes(text, spike_idx, n_frames):
    B, L, _ = spike_idx.shape
    processed_text = []

    for b in range(B):
        new_sentences = []
        carry_over = []  # stores the trailing part of the last sentence
        for l in range(L):
            sentence = text[b][l]
            words = sentence.split()
            spike = spike_idx[b, l, 0].item()
            if spike == 0:
                spike = n_frames - 2
            proportion = spike / (n_frames - 1)
            cut_point = int(round(proportion * len(words)))

            # First part to keep in current sentence
            first_part = words[:cut_point]
            # Second part to carry to next sentence
            second_part = words[cut_point:]

            # Merge with carry_over from last
            full_sentence = carry_over + first_part
            new_sentences.append(" ".join(full_sentence))

            carry_over = second_part  # prepare for next iteration
        processed_text.append(new_sentences)
    return processed_text


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
        data_loaders = data_loaders.get_alldata()
        eeg_model = model_broderick2019.Model(args)
        snn_model = SAS(args)
        trainer = trainer_broderick2019
    elif args.datasets == 'brennan2019':
        args.n_subjects = 32
        args.n_channels = 60
        args.n_slice = 1
        args.sr = 120
        args.fps = 3
        sample_t = 8.334
        data_loaders = data_brennan2019.LoadDataset(args)
        data_loaders = data_loaders.get_alldata()
        eeg_model = model_brennan2019.Model(args)
        snn_model = SAS(args)
        trainer = trainer_brennan2019

    # prepare
    # args.ckpt_ann = rf"E:\NIPS2026\ckpt\{args.datasets}\{args.model}\ann_epoch44_10@50_0.76576_10@All_0.22203.pth"
    # args.ckpt_ann = rf"E:\NIPS2026\ckpt\{args.datasets}\{args.model}+sas-brain\ann_epoch36_10@50_0.82305_10@All_0.23695.pth"
    # args.ckpt_snn = rf"E:\NIPS2026\ckpt\{args.datasets}\{args.model}+sas-brain\snn_epoch36_spike_0.02528.pth"
    scoring = make_scorer(lambda yt, yp: pearsonr(yt, yp)[0])
    ridge_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0, solver='sag'))
    cv = KFold(2, shuffle=True)

    # evaluate
    # if os.path.isfile(rf"{args.base_dir}\cache\preds.pth"):
    #     preds = torch.load(rf"{args.base_dir}\cache\preds.pth", map_location='cpu', weights_only=False)
    #     wordfs = torch.load(rf"{args.base_dir}\cache\wordfs.pth", map_location='cpu', weights_only=False)
    #     word_embs = torch.load(rf"{args.base_dir}\cache\word_embs.pth", map_location='cpu', weights_only=False)
    #     sentence_embs = torch.load(rf"{args.base_dir}\cache\sentence_embs.pth", map_location='cpu', weights_only=False)
    #     print("Data load success.")
    # else:
    trainer = trainer.Trainer({'train': data_loaders}, eeg_model, snn_model, None, None, args)
    word_encoder = spacy.load("en_core_web_lg")
    sentence_encoder = LaserEncoderPipeline(lang="eng_Latn")
    sentence_tokenizer = initialize_tokenizer(lang="eng_Latn")

    subject_preds = defaultdict(list)
    subject_wordfs = defaultdict(list)
    subject_word_embs = defaultdict(list)
    subject_sentence_embs = defaultdict(list)

    stop_loop = False
    subjects_num = 5
    for x, y, events, subjects, texts in tqdm(data_loaders):
        if stop_loop:
            break

        with torch.no_grad():
            trainer.iter += 1
            if args.ckpt_snn is not None:
                _ = trainer.snn_one_batch(x, y, events, subjects, training=False)
                spike_idx = trainer.spike_idxes[trainer.iter]  # [B, L, 1]
                x, y_sas = trainer.snn_one_batch(x, y, events, subjects, slice=True)
                texts = process_sentences_with_spikes(texts, spike_idx, trainer.n_frames)

            pred = trainer.ann(x.to(trainer.device), subjects.to(trainer.device)).detach().cpu()

            flat_text = [item for t in texts for item in t]
            flat_text = [t.replace('\n', '').replace('\r', '') for t in flat_text]
            flat_text = [t if t != '' else ' ' for t in flat_text]

            B, L = subjects.shape
            for b in range(B):
                for l in range(L):
                    subj_id = subjects[b, l].item()
                    if subj_id == subjects_num:
                        stop_loop = True
                        break  # break inner loop

                    idx = b * L + l
                    t = flat_text[idx]

                    wordf = torch.tensor([zipf(w, 'en') for w in t.split(' ')]).view(1, 1, -1)
                    wordf = F.interpolate(wordf, size=y.shape[-2], mode='linear', align_corners=False).view(-1)

                    word_emb = torch.stack([torch.tensor(we.vector) for we in word_encoder(t)]).flatten().view(1, 1, -1)
                    word_emb = F.interpolate(word_emb, size=y.shape[-2], mode='linear', align_corners=False).view(-1)

                    tokenized_sentence = sentence_tokenizer.tokenize(t)
                    sentence_emb = torch.tensor(sentence_encoder.encode_sentences([tokenized_sentence], normalize_embeddings=True)).view(1, 1, -1)
                    sentence_emb = F.interpolate(sentence_emb, size=y.shape[-2], mode='linear', align_corners=False).view(-1)

                    subject_preds[subj_id].append(pred[idx])
                    subject_wordfs[subj_id].append(wordf)
                    subject_word_embs[subj_id].append(word_emb)
                    subject_sentence_embs[subj_id].append(sentence_emb)
                if stop_loop:
                    break  # break outer loop within batch

    # Final stacking: [10, N_i, D]
    preds = torch.stack([torch.stack(subject_preds[sid]) for sid in range(subjects_num)])  # [10, N, ...]
    wordfs = torch.stack([torch.stack(subject_wordfs[sid]) for sid in range(subjects_num)])  # [10, N, ...]
    word_embs = torch.stack([torch.stack(subject_word_embs[sid]) for sid in range(subjects_num)])  # [10, N, ...]
    sentence_embs = torch.stack([torch.stack(subject_sentence_embs[sid]) for sid in range(subjects_num)])  # [10, N, ...]
    assert preds.shape[0] == wordfs.shape[0]
    cache_dir = rf"{args.base_dir}\cache"
    # torch.save(preds, rf"{cache_dir}\preds.pth")
    # torch.save(wordfs, rf"{cache_dir}\wordfs.pth")
    # torch.save(word_embs, rf"{cache_dir}\word_embs.pth")
    # torch.save(sentence_embs, rf"{cache_dir}\sentence_embs.pth")

    # n = preds.shape[0]
    # half_n = n // 4
    # indices = torch.randperm(n)[:half_n]
    # preds = preds[indices]
    # wordfs = wordfs[indices]
    # word_embs = word_embs[indices]
    # sentence_embs = sentence_embs[indices]

    # ConvLinear
    # scores = torch.zeros([preds.shape[0], 3])
    # stds = torch.zeros([preds.shape[0], 3])
    # for i in tqdm(range(preds.shape[0])):
    #     scores[i, 0], stds[i, 0] = evaluate_model(ConvLinear, preds[i], wordfs[i], epochs=700)
    #     scores[i, 1], stds[i, 1] = evaluate_model(ConvLinear, preds[i], word_embs[i], epochs=100)
    #     scores[i, 2], stds[i, 2] = evaluate_model(ConvLinear, preds[i], sentence_embs[i], epochs=100)
    #     print(scores[i])
    #     print(stds[i])

    # RidgeRegression
    scores = torch.zeros([preds.shape[0], 768, 3])
    stds = torch.zeros([preds.shape[0], 768, 3])
    for i in range(preds.shape[0]):
        if i == 0: continue
        for j in tqdm(range(768)):
            # scores[i, j, 0], stds[i, j, 0] = evaluate_model(RidgeRegression, preds[i, :, j], wordfs[i, :, j], epochs=1500)
            # scores[i, j, 1], stds[i, j, 1] = evaluate_model(RidgeRegression, preds[i, :, j], word_embs[i, :, j], epochs=1000)
            # scores[i, j, 2], stds[i, j, 2] = evaluate_model(RidgeRegression, preds[i, :, j], sentence_embs[i, :, j], epochs=700)
            score = cross_val_score(ridge_model, preds[i, :, j], wordfs[i, :, j], scoring=scoring, cv=cv)
            scores[i, j, 0], stds[i, j, 0] = np.nanmean(score), np.nanstd(score)
            score = cross_val_score(ridge_model, preds[i, :, j], word_embs[i, :, j], scoring=scoring, cv=cv)
            scores[i, j, 1], stds[i, j, 1] = np.nanmean(score), np.nanstd(score)
            score = cross_val_score(ridge_model, preds[i, :, j], sentence_embs[i, :, j], scoring=scoring, cv=cv)
            scores[i, j, 2], stds[i, j, 2] = np.nanmean(score), np.nanstd(score)
        print(torch.nanmean(scores[i], dim=0), torch.nanmean(stds[i], dim=0))
    torch.save(scores, rf"{args.base_dir}\cache\scores.pth")
    torch.save(stds, rf"{args.base_dir}\cache\stds.pth")
    scores = torch.nanmean(scores, dim=1)
    stds = torch.nanmean(stds, dim=1)

    scores = torch.nanmean(scores, dim=0)
    stds = torch.nanmean(stds, dim=0)
    print(str(scores.tolist()))
    print(str(stds.tolist()))
    with open(rf"{args.base_dir}\cache\metrics_summary.txt", "w") as f:
        f.write("Scores:\n")
        f.write(str(scores.tolist()) + "\n\n")

        f.write("Stds:\n")
        f.write(str(stds.tolist()) + "\n")