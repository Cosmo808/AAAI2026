import pdb
import torch
import torch.nn.functional as F


class MembraneLoss(torch.nn.Module):
    def __init__(self, mse=torch.nn.MSELoss(), v_decay=1, i_decay=1, alpha=0., *args, **kwargs):
        """
        :param mse: loss function
        :param v_decay: coefficient of v
        :param i_decay: coefficient of I
        :param alpha: weight of upper bound
        """
        super().__init__(*args, **kwargs)
        self.mse = mse
        self.v_decay = v_decay
        self.i_decay = i_decay
        self.alpha_value = torch.nn.Parameter(torch.tensor(alpha))

    def __call__(self, mem_seq, I, batch_idx, spike_idx, max_idx, Vth, no_spike):
        """
        :param mem_seq: membrane potential sequence (with gradient)
        :param I: current sequence (with gradient)
        :param batch_idx: global index of batch
        :param spike_idx: global index of spike
        :param Vth: threshold of membrane potential
        """
        ## monotonic-assuming loss
        # spike_mem = mem_seq[spike_idx][batch_idx]
        # target = (Vth * (spike_idx + 1) / (max_idx + 1)).to(spike_mem.device)
        # mono_loss = self.mse(spike_mem, target.unsqueeze(0))

        ## membrane loss
        if max_idx > spike_idx:
            pre_mem_v = mem_seq[spike_idx][batch_idx]
            added_I = 0
            for i in range(spike_idx + 1, max_idx + 1):
                pre_mem_v = pre_mem_v * self.v_decay + self.i_decay * I[i, batch_idx].clamp(0)
                added_I = added_I + I[i, batch_idx].clamp(0).detach()
            mem_v = pre_mem_v
        else:
            mem_v = mem_seq[max_idx][batch_idx]
            added_I = 1
        up_bound_target = (torch.tensor(Vth) * self.v_decay + self.i_decay * I[max_idx, batch_idx].detach().clamp(0)).clamp(min=Vth)
        low_bound_target = torch.tensor(Vth)
        target = self.alpha * up_bound_target + (1 - self.alpha) * low_bound_target
        mem_loss = self.mse(mem_v, target)

        if no_spike:
            mem_seq = torch.stack(mem_seq)
            mem_loss = mem_loss + self.mse(mem_seq[:, batch_idx].max().unsqueeze(0), target)

        # if added_I == 0:
        #     mem_loss = mem_loss + mono_loss
        ## negative I loss
        neg_I = I.clamp(max=0)
        I_loss = self.mse(neg_I, torch.zeros_like(neg_I))
        return mem_loss, I_loss

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_value)

class _MaskedLoss(torch.nn.Module):
    def forward(self, estimate, output, mask=None):
        feature_mask = mask.expand_as(estimate)
        estimate, output = estimate[feature_mask], output[feature_mask]
        return self._loss(estimate, output)


class L1Loss(_MaskedLoss):
    def __init__(self):
        super().__init__()
        self._loss = torch.nn.L1Loss()


class L2Loss(_MaskedLoss):
    def __init__(self):
        super().__init__()
        self._loss = torch.nn.MSELoss()


class ClipLoss(torch.nn.Module):
    """CLIP (See Open AI CLIP) constrastive loss.
    """
    def __init__(self, linear=None, twin=True, pool=False, tmin=None, tmax=None,
                 tmin_train=None, tmax_train=None, dset_args=None, center=False):
        super().__init__()
        self.linear = None
        self.pool = pool
        self.center = center
        if linear is not None:
            self.linear_est = torch.nn.LazyLinear(linear)
            if twin:
                self.linear_gt = self.linear_est
            else:
                self.linear_gt = torch.nn.LazyLinear(linear)
        self.tmin = tmin
        self.tmax = tmax
        self.tmin_train = tmin_train
        self.tmax_train = tmax_train
        self.dset_args = dset_args

    def trim_samples(self, estimates, candidates):
        """Given estimates that is [B1, C, T] and candidates
        which is [B2, C, T], return estimates_trim of size [B1, C, T']
        and candidates_trim of size [B2, C, T'], such that T'
        corresponds to the samples between [self.tmin, self.tmax]
        """
        if self.tmin is None and self.tmax is None:
            return estimates, candidates

        if self.training and (self.tmin_train is not None or self.tmax_train is not None):
            tmin, tmax = self.tmin_train, self.tmax_train
        else:
            tmin, tmax = self.tmin, self.tmax
        if (tmin is not None) or (tmax is not None):
            assert self.dset_args is not None
            assert self.dset_args.tmin is not None
            dset_tmin = self.dset_args.tmin
        if tmin is None:
            trim_min = 0
        else:
            assert tmin >= dset_tmin, 'clip.tmin should be above dset.tmin'
            trim_min = int((-dset_tmin + tmin) * self.dset_args.sample_rate)
        if tmax is None:
            trim_max = estimates.shape[-1]
        else:
            trim_max = int((-dset_tmin + tmax) * self.dset_args.sample_rate)
        estimates_trim = estimates[..., trim_min:trim_max]
        candidates_trim = candidates[..., trim_min:trim_max]
        return estimates_trim, candidates_trim

    def get_scores(self, estimates: torch.Tensor, candidates: torch.Tensor):
        """Given estimates that is [B, C, T] and candidates
        which is [B', C, T], return a [B, B'] matrix of scores of matching.
        """
        estimates, candidates = self.trim_samples(estimates, candidates)
        if self.linear:
            estimates = self.linear_est(estimates)
            candidates = self.linear_gt(candidates)
        if self.pool:
            estimates = estimates.mean(dim=2, keepdim=True)
            candidates = candidates.mean(dim=2, keepdim=True)
        if self.center:
            estimates = estimates - estimates.mean(dim=(1, 2), keepdim=True)
            candidates = candidates - candidates.mean(dim=(1, 2), keepdim=True)
        inv_norms = 1 / (1e-8 + candidates.norm(dim=(1, 2), p=2))
        # We normalize inside the einsum, to avoid creating a copy
        # of candidates, which can be pretty big.
        scores = torch.einsum("bct,oct,o->bo", estimates, candidates, inv_norms)
        return scores
    
    def get_ce_loss(self, scores):
        target = torch.arange(len(scores), device=scores.device)
        return F.cross_entropy(scores, target)

    def get_probabilities(self, estimates, candidates):
        """Given estimates that is [B, C, T] and candidates
        which is [B', C, T], return a [B, B'] matrix of probabilities of matching.
        """
        scores = self.get_scores(estimates, candidates)
        return F.softmax(scores, dim=1)

    def forward(self, estimate, candidate, mask=None):
        """Warning: estimate and candidate are not symmetrical.
        If estimate of shape [B, C, T] and candidate of size [B', C, T]
        with B'>=B, the first B samples of candidate are targets, while
        the remaining B'-B samples of candidate are only used as negatives.
        """
        assert estimate.size(0) <= candidate.size(0), "need at least as many targets as estimates"
        scores = self.get_scores(estimate, candidate)
        target = torch.arange(len(scores), device=estimate.device)
        return F.cross_entropy(scores, target)
