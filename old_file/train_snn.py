from models.snn import *
from models.simplecnn import *
from models.losses import *
import argparse
from collections import defaultdict
import time
import torch.nn.functional as F
from spikingjelly.activation_based import functional
from models.utils import *


class Trainer:
    def __init__(self, event_dataloaders, eeg_data, snn, ann, optimizer, scheduler, args):
        self.base_path = os.path.join(r"datasets", args.dataset)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.event_dataloaders = event_dataloaders
        self.eeg_data = eeg_data
        self.args = args

        self.snn = snn.to(self.device)
        self.snn.node.step_mode = 'm'
        self.ann = ann.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.k = args.k
        self.max_epoch = args.max_epoch
        self.negative_pool = None
        self.sample_t = args.sample_t
        self.n_negatives = args.n_negatives
        self.negative_pool = None
        self.cache_dir = rf".\cache\{args.dataset}"

        self.epoch = -1
        self.best_state_snn = None
        self.best_state_ann = None
        self.best_clip_loss = float('inf')
        self.best_spike_loss = float('inf')
        self.best_acc = 0
        self.best_epoch = 0
        self.early_stop_epoch = args.early_stop_epoch
        self.snn_max_epoch = args.max_epoch if args.snn_max_epoch is None else args.snn_max_epoch
        self.save_dict_path_snn = None
        self.save_dict_path_ann = None

        self.loss_snn = MembraneLoss()
        self.loss_clip = ClipLoss()
        if self.optimizer:
            self.optimizer.add_param_group({"params": self.loss_snn.alpha_value})
            self.optimizer.add_param_group({"params": self.loss_clip.parameters()})

        self.current_spike_times = []
        self.expect_spike_times = []
        self.current_spike_idxes = []
        self.expect_spike_idxes = []

        model_name = "facebook/wav2vec2-base-10k-voxpopuli"
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.w2v_model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        self.w2v_model.eval()

        if args.ckpt_snn is not None:
            self.best_state_snn = torch.load(args.ckpt_snn, map_location=self.device)
            self.snn.load_state_dict(self.best_state_snn)
            print(f"Loading snn ckpt from {args.ckpt_snn}")
        if args.ckpt_ann is not None:
            self.best_state_ann = torch.load(args.ckpt_ann, map_location=self.device)
            self.ann.load_state_dict(self.best_state_ann)
            print(f"Loading ann ckpt from {args.ckpt_ann}")

    def train(self):
        init_time = time.time()
        for epoch in range(self.max_epoch):
            self.epoch = epoch
            passed_time = (time.time() - init_time) / 60
            print(f"Epoch {epoch}|{self.max_epoch - 1}, passing {passed_time:.1f} min")

            clip_loss_train, spike_loss_train, _ = self.cycle_one_epoch(training=True)
            clip_loss_test, spike_loss_test, top_k_acc = self.cycle_one_epoch(training=False)

            if isinstance(self.scheduler, list):
                assert self.scheduler[0].name == 'snn' and self.scheduler[1].name == 'ann'
                if not self.snn_max_epoch or self.epoch < int(self.snn_max_epoch):
                    self.scheduler[0].step(spike_loss_test)
                if self.epoch >= self.args.ann_epoch:
                    self.scheduler[1].step(clip_loss_test)

            print(f"Train: {clip_loss_train:.3f}, {spike_loss_train:.3f} (clip, spike)")
            print(f"Valid: {clip_loss_test:.3f}, {spike_loss_test:.3f}, {top_k_acc:.2f}% (clip, spike, acc)\n")

            will_stop = epoch == self.max_epoch
            if self.early_stop_epoch:
                if epoch >= self.best_epoch + self.early_stop_epoch:
                    print(f"Valid loss and acc did not improve for {self.early_stop_epoch} epochs. Stopping the training.")
                    will_stop = True

            if will_stop:
                print(f"Best valid loss {self.best_clip_loss}, {self.best_spike_loss} (clip, spike) at epoch {self.best_epoch}.")
                print(f"Best top k acc {self.best_acc} at epoch {self.best_epoch}.")
                print(f"Best dict saved at {self.save_dict_path_snn} and {self.save_dict_path_ann}")
                break

    def cycle_one_epoch(self, training=True):
        if training:
            self.snn.train()
            self.ann.train()
            self.loss_clip.train()
            dataloader = self.event_dataloaders['train']
        else:
            self.ann.eval()
            self.loss_clip.eval()
            dataloader = self.event_dataloaders['test']

        if self.snn_max_epoch is None or self.epoch < int(self.snn_max_epoch):
            spike_loss = self.cycle_snn(dataloader, training)
        else:
            spike_loss = self.best_spike_loss

        if self.epoch >= self.args.ann_epoch:
            clip_loss, top_k_acc = self.cycle_ann(dataloader, training)
        else:
            clip_loss, top_k_acc = float('inf'), 0

        update_save_dict = False
        if not training:
            if clip_loss < self.best_clip_loss:
                self.best_clip_loss, self.best_epoch = clip_loss, self.epoch
                print(f"New best clip loss {clip_loss:.3f} at epoch {self.best_epoch}")
                update_save_dict = True
            if spike_loss < self.best_spike_loss:
                self.best_spike_loss, self.best_epoch = spike_loss, self.epoch
                print(f"New best spike loss {spike_loss:.3f} at epoch {self.best_epoch}")
                update_save_dict = True
            if top_k_acc > self.best_acc:
                self.best_acc, self.best_epoch = top_k_acc, self.epoch
                print(f'New best top k acc {top_k_acc:.2f}% at epoch {self.best_epoch}')
                update_save_dict = True

            if update_save_dict and self.epoch >= 0:
                self.best_state_snn = self.copy_state(self.snn.state_dict())
                self.best_state_ann = self.copy_state(self.ann.state_dict())
                self.save_dict()

        return clip_loss, spike_loss, top_k_acc

    def cycle_snn(self, dataloader, training):
        spike_loss_epoch = []

        if not training:
            self.current_spike_times = []
            self.expect_spike_times = []
            self.current_spike_idxes = []
            self.expect_spike_idxes = []

        for data_batch in tqdm(dataloader):
            subject_ids, frames, times, expect_spike_idxes, speech_reps, eegs = data_batch

            frames = frames.to(torch.float32).to(self.device)
            if self.args.dataset == 'broderick2019':
                subject_ids, runs = subject_ids[0].to(torch.int64), subject_ids[1].to(torch.int64)

            vox_events = frames.permute(1, 0, 2, 3).contiguous()
            spikes_with_gradient = self.snn(vox_events, subject_ids.to(self.device))
            current_spike_idxes = [spikes_with_gradient[:, b, 0].detach().nonzero().flatten() for b in range(self.args.bs)]
            del vox_events, spikes_with_gradient

            spike_loss = []

            for b in range(self.args.bs):
                current_spike_idx = current_spike_idxes[b]
                expect_spike_idx = expect_spike_idxes[b][expect_spike_idxes[b] != -1]

                if current_spike_idx.numel() > 5:
                    current_spike_idx = current_spike_idx[:5]

                if not training:
                    current_spike_time = times[b][current_spike_idx.cpu()]
                    expect_spike_time = times[b][expect_spike_idx.cpu()]
                    self.current_spike_times.append(current_spike_time)
                    self.expect_spike_times.append(expect_spike_time)
                    self.current_spike_idxes.append(current_spike_idx.cpu())
                    self.expect_spike_idxes.append(expect_spike_idx.cpu())

                no_spike = False
                if current_spike_idx.numel() == 0:
                    no_spike = True
                    current_spike_idx = torch.tensor([frames.shape[1] - 1], dtype=current_spike_idx.dtype, device=self.device)

                for i, exp in enumerate(expect_spike_idx):
                    try:
                        cur = current_spike_idx[i]
                    except IndexError:
                        cur = frames.shape[1] - 1
                    mem_loss, I_loss = self.loss_snn(self.snn.node.past_v, self.snn.I, b, cur, exp, self.snn.node.v_threshold, no_spike)
                    spike_loss.append(mem_loss)

            spike_loss = sum(spike_loss) / len(spike_loss)
            if training:
                self.optimizer.zero_grad()
                spike_loss.backward()
                self.optimizer.step()

            spike_loss_epoch.append(spike_loss.detach().cpu().item())
            functional.reset_net(self.snn)
            self.snn.I = []

        spike_loss_epoch = sum(spike_loss_epoch) / len(spike_loss_epoch)
        return spike_loss_epoch

    def cycle_ann(self, dataloader, training):
        clip_loss_epoch = []
        total, correct = 0, 0
        batch_id = -1

        for data_batch in tqdm(dataloader):
            batch_id += 1

            if self.epoch < int(self.snn_max_epoch):
                subject_ids, frames, times, expect_spike_idxes, _, _ = data_batch

                frames = frames.to(torch.float32).to(self.device)
                if self.args.dataset == 'broderick2019':
                    subject_ids, runs = subject_ids[0].to(torch.int64), subject_ids[1].to(torch.int64)

                # snn spiking
                vox_events = frames.permute(1, 0, 2, 3).contiguous()
                spikes_with_gradient = self.snn(vox_events, subject_ids.to(self.device))

                # obtain subject-level spiking times
                subject_spike_times = defaultdict(list)
                for b in range(self.args.bs):
                    current_spike_idx = spikes_with_gradient[:, b, 0].detach().nonzero().flatten()
                    expect_spike_idx = expect_spike_idxes[b][expect_spike_idxes[b] != -1]
                    # if current_spike_idx.numel() > expect_spike_idx.numel() + 1:
                    #     current_spike_idx = current_spike_idx[:len(expect_spike_idx)]
                    current_spike_time = times[b][current_spike_idx.cpu()]

                    if self.args.dataset == 'brennan2019':
                        subject_id = subject_ids[b].item()
                    elif self.args.dataset == 'broderick2019':
                        subject_id = (subject_ids[b].item(), runs[b].item())
                    subject_spike_times[subject_id].append(current_spike_time)

                subject_spike_times = {k: torch.cat(v).sort()[0] for k, v in subject_spike_times.items()}

                # obtain adaptive sliced data
                speech_rep_pool, eeg_pool, subjects_pool = [], [], []
                for subject_id, spike_times in subject_spike_times.items():
                    subject_eeg, sample_rate = self.eeg_data[subject_id]
                    sound_events, block_events = self.subject_events(subject_id)
                    if self.args.dataset == 'broderick2019':
                        subject_id = subject_id[0]
                    centroids = torch.tensor(block_events['start'].tolist(), device=spike_times.device)
                    distances = torch.abs(spike_times.unsqueeze(1) - centroids.unsqueeze(0))   # shape: [num_spikes, num_centroids]
                    min_distances, nearest_indices = torch.min(distances, dim=1)   # [num_spikes]

                    valid_mask = min_distances <= 1
                    valid_spike_times = spike_times[valid_mask]
                    valid_nearest_indices = nearest_indices[valid_mask]

                    centroid_groups = {}
                    for idx, centroid in enumerate(centroids):
                        centroid_groups[idx] = []

                    for spike_time, centroid_idx in zip(valid_spike_times, valid_nearest_indices):
                        centroid_groups[centroid_idx.item()].append(spike_time.item())

                    spike_times = []
                    for idx, centroid in enumerate(centroids):
                        group_spikes = centroid_groups[idx]
                        if group_spikes:
                            avg = sum(group_spikes) / len(group_spikes)
                            spike_times.append(avg)

                    for i in range(len(spike_times) - 1):
                        for j in range(i + 1, len(spike_times)):
                            start_time = spike_times[i]
                            end_time = spike_times[j]

                            if (end_time - start_time) > 2 * self.sample_t:
                                continue

                            _, rep_audio = self.wav2vec(sound_events, start_time, end_time)
                            eeg_sliced = subject_eeg[:, int(start_time * sample_rate):int(end_time * sample_rate)]
                            if rep_audio is None:
                                continue

                            speech_rep_pool.append(rep_audio.permute(0, 2, 1))
                            eeg_pool.append(eeg_sliced)
                            subjects_pool.append(subject_id)
                            
                # resample to a common temporal length
                if len(eeg_pool) == 0:
                    continue
                # common_T = sum([eeg.shape[-1] for eeg in eeg_pool]) // len(eeg_pool)
                common_T = self.sample_t * sample_rate
                for i in range(len(speech_rep_pool)):
                    # speech_rep_pool[i] = self.resample(speech_rep_pool[i], common_T)
                    speech_rep_pool[i] = F.interpolate(speech_rep_pool[i], size=common_T)
                    eeg_pool[i] = self.resample(eeg_pool[i], common_T)

                speech_rep_pool = torch.cat(speech_rep_pool, dim=0).float().to(self.device)  # [n, 768, T]
                eeg_pool = torch.cat(eeg_pool, dim=0).float().to(self.device)  # [n, 60, T]
                subjects_pool = torch.tensor(subjects_pool).to(self.device)  # [n]
                assert speech_rep_pool.dim() == 3 and speech_rep_pool.dim() == 3

                if self.epoch == int(self.snn_max_epoch) - 1:
                    pools = {'speech_rep': speech_rep_pool, 'eeg': eeg_pool, 'subjects': subjects_pool}
                    pools_path = rf"{self.cache_dir}\batch{batch_id}_pools_train.pt" if training else rf"{self.cache_dir}\batch{batch_id}_pools_test.pt"
                    os.makedirs(os.path.dirname(pools_path), exist_ok=True)
                    torch.save(pools, f=pools_path)

            else:
                pools_path = rf"{self.cache_dir}\batch{batch_id}_pools_train.pt" if training else rf"{self.cache_dir}\batch{batch_id}_pools_test.pt"
                pools = torch.load(pools_path, weights_only=False, map_location=self.device)
                speech_rep_pool = pools['speech_rep']
                eeg_pool = pools['eeg']
                subjects_pool = pools['subjects']

            # ann constrastive learning
            eeg_rep_pool = self.ann(eeg_pool, subjects_pool)  # [n, 768, T]
            if training:
                if self.n_negatives is not None:
                    if self.negative_pool is None:
                        self.negative_pool = speech_rep_pool
                        candidate = speech_rep_pool
                    else:
                        kept = torch.randperm(self.negative_pool.shape[0])
                        self.negative_pool = self.negative_pool[kept]
                        self.negative_pool = self.negative_pool[:self.n_negatives, ...].clone()
                        candidate = torch.cat((speech_rep_pool, self.negative_pool), dim=0)
                        self.negative_pool = torch.cat((speech_rep_pool, self.negative_pool), dim=0)
                else:
                    candidate = speech_rep_pool
                clip_loss = self.loss_clip(eeg_rep_pool, candidate)
                self.optimizer.zero_grad()
                clip_loss.backward()
                self.optimizer.step()
                with torch.no_grad():
                    clip_loss = self.loss_clip(eeg_rep_pool, speech_rep_pool)
            else:
                with torch.no_grad():
                    scores = self.loss_clip.get_scores(eeg_rep_pool, speech_rep_pool)
                    clip_loss = self.loss_clip.get_ce_loss(scores)
                try:
                    top_k_preds = scores.topk(int(eeg_rep_pool.shape[0] / self.k), dim=1).indices
                    ground_truth = torch.arange(scores.size(0), device=self.device).view(-1, 1)
                    correct += (top_k_preds == ground_truth).sum().item()
                    total += scores.size(0)
                except RuntimeError:
                    pass

            functional.reset_net(self.snn)
            self.snn.I = []
            clip_loss_epoch.append(clip_loss.detach().cpu().item())

        clip_loss_epoch = sum(clip_loss_epoch) / len(clip_loss_epoch)
        top_k_acc = correct / total * 100 if total != 0 else 0
        return clip_loss_epoch, top_k_acc

    def subject_pred(self):
        self.ann.eval()
        self.loss_clip.eval()
        dataloader = self.event_dataloaders['test']
        load_batch = os.path.exists(self.cache_dir)

        subject_correct = defaultdict(int)
        subject_total = defaultdict(int)

        batch_id = -1
        for data_batch in tqdm(dataloader):
            batch_id += 1

            if not load_batch:
                subject_ids, frames, times, expect_spike_idxes, _, _ = data_batch

                frames = frames.to(torch.float32).to(self.device)
                if self.args.dataset == 'broderick2019':
                    subject_ids, runs = subject_ids[0].to(torch.int64), subject_ids[1].to(torch.int64)

                vox_events = frames.permute(1, 0, 2, 3).contiguous()
                spikes_with_gradient = self.snn(vox_events, subject_ids.to(self.device))

                subject_spike_times = defaultdict(list)
                for b in range(self.args.bs):
                    current_spike_idx = spikes_with_gradient[:, b, 0].detach().nonzero().flatten()
                    expect_spike_idx = expect_spike_idxes[b][expect_spike_idxes[b] != -1]
                    # if current_spike_idx.numel() > expect_spike_idx.numel() + 1:
                    #     current_spike_idx = current_spike_idx[:len(expect_spike_idx)]
                    current_spike_time = times[b][current_spike_idx.cpu()]

                    if self.args.dataset == 'brennan2019':
                        subject_id = subject_ids[b].item()
                    elif self.args.dataset == 'broderick2019':
                        subject_id = (subject_ids[b].item(), runs[b].item())
                    subject_spike_times[subject_id].append(current_spike_time)

                subject_spike_times = {k: torch.cat(v).sort()[0] for k, v in subject_spike_times.items()}

                speech_rep_pool, eeg_pool, subjects_pool = [], [], []
                for subject_id, spike_times in subject_spike_times.items():
                    subject_eeg, sample_rate = self.eeg_data[subject_id]
                    sound_events, block_events = self.subject_events(subject_id)
                    if self.args.dataset == 'broderick2019':
                        subject_id = subject_id[0]
                    centroids = torch.tensor(block_events['start'].tolist(), device=spike_times.device)
                    distances = torch.abs(spike_times.unsqueeze(1) - centroids.unsqueeze(0))  # shape: [num_spikes, num_centroids]
                    min_distances, nearest_indices = torch.min(distances, dim=1)  # [num_spikes]

                    valid_mask = min_distances <= 1
                    valid_spike_times = spike_times[valid_mask]
                    valid_nearest_indices = nearest_indices[valid_mask]

                    centroid_groups = {}
                    for idx, centroid in enumerate(centroids):
                        centroid_groups[idx] = []

                    for spike_time, centroid_idx in zip(valid_spike_times, valid_nearest_indices):
                        centroid_groups[centroid_idx.item()].append(spike_time.item())

                    spike_times = []
                    for idx, centroid in enumerate(centroids):
                        group_spikes = centroid_groups[idx]
                        if group_spikes:
                            avg = sum(group_spikes) / len(group_spikes)
                            spike_times.append(avg)

                    for i in range(len(spike_times) - 1):
                        for j in range(i + 1, len(spike_times)):
                            start_time = spike_times[i]
                            end_time = spike_times[j]

                            if (end_time - start_time) > 2 * self.sample_t:
                                continue

                            _, rep_audio = self.wav2vec(sound_events, start_time, end_time)
                            eeg_sliced = subject_eeg[:, int(start_time * sample_rate):int(end_time * sample_rate)]
                            if rep_audio is None:
                                continue

                            speech_rep_pool.append(rep_audio.permute(0, 2, 1))
                            eeg_pool.append(eeg_sliced)
                            subjects_pool.append(subject_id)

                if len(eeg_pool) == 0:
                    continue
                common_T = self.sample_t * sample_rate
                for i in range(len(speech_rep_pool)):
                    speech_rep_pool[i] = F.interpolate(speech_rep_pool[i], size=common_T)
                    eeg_pool[i] = self.resample(eeg_pool[i], common_T)

                speech_rep_pool = torch.cat(speech_rep_pool, dim=0).float().to(self.device)  # [n, 768, T]
                eeg_pool = torch.cat(eeg_pool, dim=0).float().to(self.device)  # [n, 60, T]
                subjects_pool = torch.tensor(subjects_pool).to(self.device)  # [n]
                assert speech_rep_pool.dim() == 3 and speech_rep_pool.dim() == 3

            else:
                pools_path = rf"{self.cache_dir}\batch{batch_id}_pools_test.pt"
                pools = torch.load(pools_path, weights_only=False, map_location=self.device)
                speech_rep_pool = pools['speech_rep']
                eeg_pool = pools['eeg']
                subjects_pool = pools['subjects']

            with torch.no_grad():
                eeg_rep_pool = self.ann(eeg_pool, subjects_pool)  # [n, 768, T]
                scores = self.loss_clip.get_scores(eeg_rep_pool, speech_rep_pool)
            try:
                top_k_preds = scores.topk(int(eeg_rep_pool.shape[0] / self.k), dim=1).indices
                ground_truth = torch.arange(scores.size(0), device=self.device).view(-1, 1)

                for subj in subjects_pool.unique():
                    mask = (subjects_pool == subj)
                    subj_indices = mask.nonzero(as_tuple=True)[0]

                    if subj_indices.numel() > 0:
                        subj_top_k_preds = top_k_preds[subj_indices]
                        subj_ground_truth = ground_truth[subj_indices]

                        # Compute correct predictions for this subject
                        subject_correct[subj.item()] += (subj_top_k_preds == subj_ground_truth).sum().item()
                        subject_total[subj.item()] += subj_indices.numel()
            except RuntimeError:
                pass

            functional.reset_net(self.snn)
            self.snn.I = []

        subject_acc = {subj: subject_correct[subj] / subject_total[subj] * 100 for subj in subject_correct}
        return subject_acc

    def resample(self, rep, sample_num):
        assert rep.dim() == 2
        if rep.shape[-1] == sample_num:
            return rep.unsqueeze(0)

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
        if interpolated_rep.dim() == 2:
            interpolated_rep = interpolated_rep.unsqueeze(0)
        return interpolated_rep

    def wav2vec(self, sound_events, start: float, stop: float):
        sound_start = np.array(sound_events['start'].tolist())
        index = (sound_start > start).argmax()
        index -= 1
        filepath = (sound_events.iloc[index])['filepath']
        start -= sound_start[index]
        stop -= sound_start[index]

        try:
            wav, sr = self.extract_wav(filepath, start, stop)
        except AssertionError:
            return None, None

        wav = torch.mean(wav, dim=0)
        model_sr = self.feature_extractor.sampling_rate
        wav = julius.resample.ResampleFrac(old_sr=int(sr), new_sr=model_sr)(wav)

        # [1, T]
        processed_wav = self.feature_extractor(wav, return_tensors="pt", sampling_rate=model_sr, do_normalize=True).input_values
        with torch.no_grad():
            outputs = self.w2v_model(processed_wav.to(self.device), output_hidden_states=True)
        hidden_states = outputs.get("hidden_states")
        last_hidden_state = outputs.get("last_hidden_state")
        if isinstance(hidden_states, tuple):
            hidden_states = torch.stack(hidden_states)
        # hidden_states[0] is equal to last_hidden_state
        return hidden_states, last_hidden_state

    def extract_wav(self, filepath, onset: float, offset: float):
        try:
            info = torchaudio.info(str(filepath))
            sr = float(info.sample_rate)
        except RuntimeError:
            with sf.SoundFile(filepath) as f:
                sr = float(f.samplerate)
        frame_offset = np.round(onset * sr).astype(int) if isinstance(onset, np.ndarray) else int(round(onset * sr))
        num_frames = np.round((offset - onset) * sr).astype(int) if isinstance((offset - onset), np.ndarray) else int(
            round((offset - onset) * sr))
        wav = torchaudio.load(filepath, frame_offset=frame_offset, num_frames=num_frames)[0]
        delta = abs(wav.shape[-1] / sr - offset + onset)
        assert delta <= 0.1, (delta, filepath, onset, offset, onset - offset)
        return wav, sr

    def subject_events(self, subject_id):
        files = os.listdir(self.base_path)
        if self.args.dataset == 'brennan2019':
            subject_idxes = [file for file in files if "S" in file]
            subject_idx = subject_idxes[subject_id]
            event_path = os.path.join(self.base_path, subject_idx, 'events.csv')
        elif self.args.dataset == 'broderick2019':
            subject_idxes = [file for file in files if "run" in file]
            subject_idxes = np.array([subject_idx.split('_')[0] for subject_idx in subject_idxes])
            subject_idxes = np.unique(subject_idxes).tolist()
            subject_idx = subject_idxes[subject_id[0]]
            event_path = os.path.join(self.base_path, f"{subject_idx}_run{subject_id[1] + 1}", 'events.csv')

        events = pd.read_csv(event_path)
        sound_events = events[events['kind'] == 'sound']
        block_events = events[events['kind'] == 'block']
        sound_events.reset_index(drop=True, inplace=True)
        block_events.reset_index(drop=True, inplace=True)
        return sound_events, block_events

    def save_dict(self):
        if self.save_dict_path_snn is not None:
            if os.path.exists(self.save_dict_path_snn):
                os.remove(self.save_dict_path_snn)
        if self.save_dict_path_ann is not None:
            if os.path.exists(self.save_dict_path_ann):
                os.remove(self.save_dict_path_ann)

        self.save_dict_path_snn = rf".\ckpt\snn_dict_n{self.args.n_frames}_s{self.args.stride}_epoch{self.epoch}.pt"
        self.save_dict_path_ann = rf".\ckpt\ann_dict_n{self.args.n_frames}_s{self.args.stride}_epoch{self.epoch}.pt"
        os.makedirs(os.path.dirname(self.save_dict_path_snn), exist_ok=True)
        os.makedirs(os.path.dirname(self.save_dict_path_ann), exist_ok=True)

        torch.save(self.best_state_snn, f=self.save_dict_path_snn)
        torch.save(self.best_state_ann, f=self.save_dict_path_ann)

        try:
            spikes = {
                'current_times': torch.cat(self.current_spike_times),
                'current_idxes': torch.cat(self.current_spike_idxes),
                'expect_times': torch.cat(self.expect_spike_times),
                'expect_idxes': torch.cat(self.expect_spike_idxes),
            }
            spikes_path = r".\ckpt\spikes.pt"
            os.makedirs(os.path.dirname(spikes_path), exist_ok=True)
            torch.save(spikes, f=spikes_path)
        except RuntimeError:
            pass

    @staticmethod
    def copy_state(state):
        return {k: v.cpu().clone() for k, v in state.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='broderick2019')  # brennan2019  broderick2019
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--early_stop_epoch', type=int, default=20)
    parser.add_argument('--n_negatives', type=int, default=None)
    parser.add_argument('--ckpt_snn', type=str, default=None)
    parser.add_argument('--ckpt_ann', type=str, default=None)
    parser.add_argument('--k', type=float, default=5)
    parser.add_argument('--n_frames', type=int, default=50)
    parser.add_argument('--stride', type=int, default=10)
    parser.add_argument('--ann_epoch', type=int, default=-1)
    parser.add_argument('--snn_max_epoch', default=None)
    args = parser.parse_args()
    if args.dataset == 'brennan2019':
        from data_process.brennan2019.brennan2019_event import *
        n_subjects = 32
        args.sample_t = 8
    elif args.dataset == 'broderick2019':
        from data_process.broderick2019.broderick2019_event import *
        n_subjects = 19
        args.sample_t = 5

    path = rf"datasets\{args.dataset}"
    eeg_data = torch.load(rf"{path}\eeg.pt", map_location='cpu', weights_only=False)
    event_dataloaders = generate_event_dataloader(path, args.bs, args.n_frames, args.stride, evaluate=False)
    train_num, test_num = 0, 0
    for data_batch in event_dataloaders['train']:
        subject_ids, frames, _, _, speech_reps, eegs = data_batch
        resolution = frames.shape[3:]  # frame [bs, n_frames, 2, n_electrodes]
        in_channels = eegs.shape[-2]
        feature_dim = speech_reps.shape[-2]
        train_num += frames.shape[0]
    for data_batch in event_dataloaders['test']:
        subject_ids, frames, _, _, speech_reps, eegs = data_batch
        test_num += frames.shape[0]
    print(f"Sample numbers {train_num}/{test_num} (train/test)")

    snn_model = SAS(resolution, n_subjects)
    ann = SimpleConv(in_channels, 240, 1, feature_dim, n_subjects)

    optimizer = torch.optim.AdamW([
            {"params": snn_model.parameters(), "lr": args.lr, "name": "snn"},
            {"params": ann.parameters(), "lr": args.lr, "name": "ann"}
        ],
        betas=(0.9, 0.999),
        weight_decay=0.001,
    )

    scheduler_snn = GroupReduceLROnPlateau(optimizer, group_index=0, mode='min', factor=0.3, patience=3, min_lr=1e-6, verbose=True, name='snn')
    scheduler_ann = GroupReduceLROnPlateau(optimizer, group_index=1, mode='min', factor=0.3, patience=5, min_lr=1e-6, verbose=True, name='ann')
    schedulers = [scheduler_snn, scheduler_ann]

    print(f"The snn model contains {sum(p.numel() for p in snn_model.parameters() if p.requires_grad)} parameters.")
    print(f"The ann contains {sum(p.numel() for p in ann.parameters() if p.requires_grad)} parameters.")

    trainer = Trainer(event_dataloaders, eeg_data, snn_model, ann, optimizer, schedulers, args)
    trainer.train()
