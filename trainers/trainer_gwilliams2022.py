import pdb
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from torch.profiler import profile, record_function, ProfilerActivity
from timeit import default_timer as timer
import numpy as np
import copy
import os
import gc
import faiss
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, cohen_kappa_score
from models.losses import *
from einops import rearrange
from spikingjelly.activation_based import functional
from scipy.interpolate import splrep, splev
from scipy.signal import resample_poly


class Trainer(object):
    def __init__(self, data_loaders, ann, snn, optimizer, scheduler, args):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.snn = snn.to(self.device)
        self.ann = ann.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args

        self.data_loaders = data_loaders
        self.criterion_ann = ClipLoss()
        self.criterion_snn = MembraneLoss()

        self.iter = -1
        self.epoch = -1
        self.best_epoch = 0
        self.best_state_ann = None
        self.best_state_snn = None
        self.save_dir_snn = None
        self.save_dir_ann = None

        self.n_frames = None
        self.downstream_metric = None
        self.expect_spike_idxes = None
        self.spike_idxes = None
        self.MCMC_init(mode='max')

        if args.ckpt_snn is not None:
            self.best_state_snn = torch.load(args.ckpt_snn, map_location=self.device)
            self.snn.load_state_dict(self.best_state_snn)
            print(f"Loading snn ckpt from {args.ckpt_snn}")
        if args.ckpt_ann is not None:
            self.best_state_ann = torch.load(args.ckpt_ann, map_location=self.device)
            self.ann.load_state_dict(self.best_state_ann, strict=False)
            print(f"Loading ann ckpt from {args.ckpt_ann}")

        self.negative_pool = None
        self.candidate = []
        try:
            for data_batch in self.data_loaders['test']:
                self.candidate.append(data_batch[1])
            self.candidate = torch.cat(self.candidate, dim=0).float()
            self.candidate = rearrange(self.candidate, 'A L C T -> (A L) C T')
        except KeyError:
            pass

    def ann_one_batch(self, x, y, events, subjects, training):
        B, L, C, T = x.shape

        with torch.no_grad():
            x_sas, y_sas = self.snn_one_batch(x, y, events, subjects, slice=True)   # [B, 5, 60/768, 1000] on cuda

        pred = self.ann(x_sas, subjects.to(self.device))   # [B * 5, 768, 1000]
        y_sas = rearrange(y_sas, 'B L C T -> (B L) C T')

        if training:
            if self.args.n_negatives is not None:
                if self.negative_pool is None:
                    self.negative_pool = y_sas
                    candidate = y_sas
                else:
                    kept = torch.randperm(self.negative_pool.shape[0])[:self.args.n_negatives]
                    self.negative_pool = self.negative_pool[kept]
                    candidate = torch.cat((y_sas, self.negative_pool), dim=0)
                    self.negative_pool = torch.cat((y_sas, self.negative_pool), dim=0)
            else:
                candidate = y_sas

            scores = self.criterion_ann.get_scores(pred, candidate.float().to(self.device))
            loss = self.criterion_ann.get_ce_loss(scores)
            self.optimizer.zero_grad()
            loss.backward()
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.ann.parameters(), self.args.grad_clip)
            self.optimizer.step()
            assert self.scheduler[0].name == 'ann'
            self.scheduler[0].step()

            scores = F.softmax(scores.detach(), dim=-1)
            accuracy_per_sample = torch.diagonal(scores, dim1=0, dim2=1)
            accuracy_per_sample = rearrange(accuracy_per_sample, '(B L) -> B L', B=B, L=L).mean(dim=1)  # size: [B]
            if self.epoch > 0:
                self.MCMC_step(accuracy_per_sample, mode='max')
            self.downstream_metric[self.iter] = accuracy_per_sample

            return loss.detach().cpu().item()

        else:
            assert y_sas.shape[0] <= 50
            total = y_sas.shape[0] + self.candidate.shape[0]
            C, T = y_sas.shape[1], y_sas.shape[2]

            candidate_all = torch.empty((total, C, T), device=self.device, dtype=y_sas.dtype)
            candidate_50 = torch.empty((50, C, T), device=self.device, dtype=y_sas.dtype)
            candidate_all[:y_sas.shape[0]] = y_sas
            candidate_50[:y_sas.shape[0]] = y_sas

            candidate_all[y_sas.shape[0]:].copy_(self.candidate, non_blocking=True)
            rest_50 = torch.randperm(self.candidate.shape[0])[:(50 - y_sas.shape[0])]
            candidate_50[y_sas.shape[0]:].copy_(self.candidate[rest_50], non_blocking=True)

            ground_truth = torch.arange(pred.shape[0], device=self.device).view(-1, 1)

            scores_all = self.criterion_ann.get_scores(pred, candidate_all)
            topk_all = scores_all.topk(k=10, dim=1, sorted=False).indices
            correct_all = (topk_all == ground_truth).sum(dim=1).tolist()

            scores_50 = self.criterion_ann.get_scores(pred, candidate_50)
            topk_50 = scores_50.topk(k=10, dim=1, sorted=False).indices
            correct_50 = (topk_50 == ground_truth).sum(dim=1).tolist()

            return correct_all, correct_50

    def snn_one_batch(self, x, y, events, subjects, training=False, slice=False):
        # x: [B, 20, 6, 6000]
        # expect_idxes: [B, L]
        B, L, C, T = x.shape

        if slice:
            if self.args.frozen_snn:
                return x.to(self.device), y.to(self.device)
            x = rearrange(x, 'B L C T -> B C (L T)', L=L, T=T).to(self.device)
            y = rearrange(y, 'B L C T -> B C (L T)', L=L, T=T).to(self.device)
            x_sas = []
            y_sas = []
            for b in range(B):
                spike_idx = self.spike_idxes[self.iter, b]   # [L, ]
                # spike_idx = [self.n_frames if len(s) == 0 else s[0].item() + 1 for s in spike_idx]
                flat_list = [s.item() + 1 + i * self.n_frames for i, row in enumerate(spike_idx) for s in row if s != -1]
                sorted_list = sorted(flat_list)
                spike_idx = [0] + sorted_list if sorted_list[0] != 0 else sorted_list
                spike_time = torch.tensor(spike_idx) / self.args.fps * self.args.sr
                spike_time = spike_time.to(torch.int64)
                x_sas.append(torch.stack(
                    [self.resample_F(x[b, :, spike_time[i]:spike_time[i + 1]], sample_num=T) for i in range(len(spike_time) - 1)]
                ))
                y_sas.append(torch.stack(
                    [self.resample_F(y[b, :, spike_time[i]:spike_time[i + 1]], sample_num=T) for i in range(len(spike_time) - 1)]
                ))
            x_sas = torch.stack(x_sas).float()
            y_sas = torch.stack(y_sas).float()
            return x_sas, y_sas  # [B, l, C, T]

        events = rearrange(events, 'B L t P C -> (B L) t P C', B=B, L=L)
        spike_idxes = self.snn(events.to(self.device))
        expect_idxes = self.expect_spike_idxes[self.iter].to(torch.int64)
        expect_idxes = rearrange(expect_idxes, 'B L k -> (B L) k', L=L)
        spike_loss = []
        for b in range(B * L):
            spike_idx = spike_idxes[b]
            expect_idx = expect_idxes[b]
            expect_idx = expect_idx[expect_idx != -1]

            no_spike = True if spike_idx.numel() == 0 else False
            expect_idx = expect_idx.unsqueeze(0) if expect_idx.ndim == 0 else expect_idx
            if spike_idx.numel() >= self.expect_spike_idxes.shape[-1]:
                spike_idx = spike_idx[:self.expect_spike_idxes.shape[-1]]
            else:
                padded_idxes = torch.full((self.expect_spike_idxes.shape[-1],), - 1, dtype=spike_idx.dtype)
                padded_idxes[:spike_idx.size(0)] = spike_idx
                spike_idx = padded_idxes

            for i, exp in enumerate(expect_idx):
                s = spike_idx[i] if spike_idx[i] != -1 else self.n_frames
                mem_loss, I_loss = self.criterion_snn(self.snn.snn.node.past_v, self.snn.snn.I, b, s, exp, self.snn.snn.node.v_threshold, no_spike)
                spike_loss.append(mem_loss)

            if no_spike:
                spike_idx = torch.full((self.expect_spike_idxes.shape[-1],), - 1, dtype=spike_idx.dtype)
                spike_idx[0] = torch.sort(torch.randperm(self.n_frames)[0])[0] if not self.args.frozen_snn else self.n_frames - 1
            self.spike_idxes[self.iter, b // L, b % L] = spike_idx.cpu()

        spike_loss = sum(spike_loss) / len(spike_loss)

        if training:
            self.optimizer.zero_grad()
            spike_loss.backward()
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.snn.parameters(), self.args.grad_clip)
            self.optimizer.step()
            assert self.scheduler[1].name == 'snn'
            self.scheduler[1].step()

        functional.reset_net(self.snn.snn)
        self.snn.snn.I = []
        return spike_loss.detach().cpu().item()

    def run_one_epoch(self, mode):
        self.iter = -1
        if mode == 'train':
            torch.cuda.empty_cache()
            self.ann.train()
            self.snn.train()
            spike_losses = []
            losses = []
            for x, y, events, subjects in tqdm(self.data_loaders[mode]):
                self.iter += 1

                if self.args.frozen_snn:
                    spike_losses.append(0)
                else:
                    spike_loss = self.snn_one_batch(x, y, events, subjects, training=True)
                    spike_losses.append(spike_loss)

                if self.args.frozen_ann:
                    losses.append(0)
                else:
                    loss = self.ann_one_batch(x, y, events, subjects, training=True)
                    losses.append(loss)

            return losses, spike_losses
        else:
            if self.epoch % 10 != 9: return 0, 0, 0
            torch.cuda.empty_cache()
            self.ann.eval()
            corrects10_50, corrects10_all = [], []
            for x, y, events, subjects in tqdm(self.data_loaders[mode]):
                self.iter += 1
                spike_loss = self.snn_one_batch(x, y, events, subjects, training=False)
                correct_all, correct_50 = self.ann_one_batch(x, y, events, subjects, training=False)
                corrects10_50 += correct_50
                corrects10_all += correct_all

            top10_50 = sum(corrects10_50) / len(corrects10_50)
            top10_all = sum(corrects10_all) / len(corrects10_all)
            return top10_50, top10_all, spike_loss

    def train(self):
        start_time = timer()
        top10_50_best = 0
        top10_all_best = 0
        spike_best = np.inf
        for epoch in range(self.args.max_epoch):
            self.epoch = epoch
            losses, spike_losses = self.run_one_epoch(mode='train')
            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                top10_50, top10_all, spike_loss = self.run_one_epoch(mode='test')

                print(
                    "Epoch {}/{} | training loss: {:.2f}/{:.5f}, top10@50: {:.5f}, top10@All: {:.5f}, LR: {:.2e}, elapsed {:.1f} mins".format(
                        epoch, self.args.max_epoch, np.mean(losses), np.mean(spike_losses), top10_50, top10_all,
                        optim_state['param_groups'][0]['lr'], (timer() - start_time) / 60)
                )
                if top10_50 > top10_50_best:
                    print("topk acc increasing....saving weights !! ")
                    print(
                        "Val Evaluation: top10@50: {:.5f}, top10@All: {:.5f}, spike_loss: {:.3f}".format(
                            top10_50, top10_all, spike_loss)
                    )
                    self.best_epoch = epoch
                    top10_50_best = top10_50
                    top10_all_best = top10_all
                    spike_best = spike_loss
                    self.best_state_ann = copy.deepcopy(self.ann.state_dict())
                    self.best_state_snn = copy.deepcopy(self.snn.state_dict())
                    self.save_dict((top10_50, top10_all, spike_loss))
                print(f"Epoch {epoch}/{self.args.max_epoch} fnished...\n\n")

        self.ann.load_state_dict(self.best_state_ann)
        self.snn.load_state_dict(self.best_state_snn)
        with torch.no_grad():
            print("***************************Test results************************")
            torch.cuda.empty_cache()
            self.candidate = []
            for data_batch in self.data_loaders['test']:
                self.candidate.append(data_batch[1])
            self.candidate = torch.cat(self.candidate, dim=0).float()
            self.candidate = rearrange(self.candidate, 'A L C T -> (A L) C T').to(self.device)
            top10_50, top10_all, spike_loss = self.run_one_epoch(mode='test')
            print(
                "Test Evaluation: top10@50: {:.5f}, top10@All: {:.5f}, spike_loss{:.5f}".format(
                    top10_50, top10_all, spike_loss)
            )
            self.epoch += 1
            self.save_dict((top10_50, top10_all, spike_loss))

    def save_dict(self, values):
        if self.epoch == 0:
            return

        if self.save_dir_ann is not None:
            if os.path.exists(self.save_dir_ann):
                os.remove(self.save_dir_ann)
        if self.save_dir_snn is not None:
            if os.path.exists(self.save_dir_snn):
                os.remove(self.save_dir_snn)

        top10_50, top10_all, spike_loss = values
        self.save_dir_ann = self.args.save_dir + r"\ann_epoch{}_10@50_{:.5f}_10@All_{:.5f}.pth".format(self.best_epoch, top10_50, top10_all)
        self.save_dir_snn = self.args.save_dir + r"\snn_epoch{}_spike_{:.5f}.pth".format(self.best_epoch, spike_loss)
        os.makedirs(os.path.dirname(self.save_dir_ann), exist_ok=True)
        os.makedirs(os.path.dirname(self.save_dir_snn), exist_ok=True)

        torch.save(self.best_state_ann, f=self.save_dir_ann)
        torch.save(self.best_state_snn, f=self.save_dir_snn)
        print("ann model save in " + self.save_dir_ann)
        print("snn model save in " + self.save_dir_snn)

    def MCMC_init(self, mode):
        for data_batch in self.data_loaders['train']:
            B, L, C, T = data_batch[0].shape  # B, 5, 208, 1200
            break
        duration = T / self.args.sr  # 10 seconds
        self.n_frames = round(duration * self.args.fps)  # 20 frames
        # expect_spike_idxes = torch.ones(size=[len(self.data_loaders['train']), B, L]) * (self.n_frames - 1)
        self.expect_spike_idxes = torch.stack([
            torch.sort(torch.randperm(self.n_frames)[:self.args.n_slice])[0]
            for _ in range(len(self.data_loaders['train']) * B * L)
        ]).view(len(self.data_loaders['train']), B, L, self.args.n_slice)
        self.spike_idxes = torch.zeros_like(self.expect_spike_idxes) - 1
        self.spike_idxes[:, :, :, 0] = self.n_frames - 1
        self.downstream_metric = torch.zeros(size=[len(self.data_loaders['train']), B], device=self.device)
        if mode == 'min':
            self.downstream_metric += np.inf

    def MCMC_step(self, metric, mode):    # MCMC
        accept = []
        for b in range(self.args.bs):
            u = torch.rand(1).item()
            if mode == 'min':
                p = self.downstream_metric[self.iter, b] / metric[b]
            elif mode == 'max':
                p = metric[b] / self.downstream_metric[self.iter, b]

            if u < p:  # accept
                accept.append(b)
                self.expect_spike_idxes[self.iter, b] = self.spike_idxes[self.iter, b]
        # print(f"Accept {len(accept)}/{self.args.bs} batches for new slicing labels for iter {self.iter} ")

    def resample(self, rep, sample_num):
        assert rep.dim() == 2
        if rep.shape[-1] == sample_num:
            return rep

        if rep.shape[-1] < sample_num:  # upsample
            features, time = rep.shape
            x_original = np.linspace(0, 1, time)
            x_target = np.linspace(0, 1, sample_num)
            interpolated_rep = np.zeros((features, sample_num))
            for j in range(features):
                tck = splrep(x_original, rep[j, :].cpu().numpy(), k=3, s=0)
                interpolated_rep[j, :] = splev(x_target, tck)
        else:  # downsample
            interpolated_rep = resample_poly(rep, up=sample_num, down=rep.shape[-1], axis=1)

        interpolated_rep = torch.tensor(interpolated_rep)
        return interpolated_rep

    def resample_F(self, rep: torch.Tensor, sample_num: int) -> torch.Tensor:
        assert rep.dim() == 2  # [features, time]
        features, time = rep.shape
        if time == sample_num:
            return rep

        rep = rep.unsqueeze(0)  # [1, features, time]
        rep_interp = F.interpolate(rep, size=sample_num, mode='linear', align_corners=True)
        return rep_interp.squeeze(0)