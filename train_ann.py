import pdb
from einops import rearrange
from tqdm import tqdm
from models import model_broderick2019, model_brennan2019
from models.simplecnn import *
from models.losses import *
from torch import nn
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
import time


class Trainer:
    def __init__(self, dataloaders, eeg_model, args):
        self.length = args.length
        self.dataloaders = dataloaders
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = eeg_model.float().to(self.device)
        for m in self.model.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, betas=(0.9, 0.999))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=0.001  # Start with 0.01 or 0.001 (tune this!)
        )
        print(f"The model contains {sum(p.numel() for p in self.model.parameters() if p.requires_grad)} parameters.")

        self.args = args
        self.pca = args.pca
        self.load_batch = args.load_batch
        self.k = args.k
        self.max_epoch = args.max_epoch
        self.n_negatives = args.n_negatives
        self.epoch = 0
        self.best_state: tp.Optional[dict] = None
        self.best_loss = float('inf')
        self.best_acc = 0
        self.best_epoch = 0
        self.best_acc_epoch = 0
        self.last_test_epoch = 0
        self.early_stop_epoch = args.early_stop_epoch
        self.eval_every_epcoh = args.eval_every_epcoh
        self.save_dict_path = None

        self.loss = self.create_loss(args.loss).to(self.device)
        self.train_losses, self.test_losses, self.accs = [], [], []
        self.negative_pool = None

        if args.ckpt is not None:
            self.ckpt = args.ckpt
            self.best_state = torch.load(args.ckpt, map_location=self.device)
            self.model.load_state_dict(self.best_state)
            print(f"Loading ckpt from {args.ckpt}")

    def train(self):
        init_time = time.time()
        for epoch in range(self.max_epoch):
            self.epoch = epoch
            train_loss, _ = self.run_one_epoch(training=True)
            test_loss, top_k_acc = self.run_one_epoch(training=False)
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            self.accs.append(top_k_acc)
            # if epoch % 10 == 5:
                # self.plot_process()
            passed_time = (time.time() - init_time) / 60
            print(f"Epoch {epoch}|{self.max_epoch - 1}: loss {train_loss:.3f} / {test_loss:.3f} (train / test), "
                  f"acc {top_k_acc:.2f}% (test), passing {passed_time:.1f} min")

            will_stop = epoch == self.max_epoch
            if self.early_stop_epoch:
                if epoch >= self.best_epoch + self.early_stop_epoch and epoch >= self.best_acc_epoch + self.early_stop_epoch:
                    logger.warning("Valid loss and acc did not improve for "
                                   f"{self.early_stop_epoch} epochs. "
                                   "Stopping the training.")
                    will_stop = True
                    
            # if epoch % self.eval_every_epcoh == 0 or will_stop:
            #     if self.best_epoch > self.last_test_epoch:
            #         assert self.best_state is not None
            #         self.model.load_state_dict(self.best_state)
            #         self.last_test_epoch = epoch
            #         print(f"Loading the best state dict to the current model.")

            if will_stop:
                print(f"Best valid loss {self.best_loss} at epoch {self.best_epoch}.")
                print(f"Best top k acc {self.best_acc} at epoch {self.best_acc_epoch}.")
                print(f"Best dict saved at {self.save_dict_path}")
                # self.plot_process()
                break

    def run_one_epoch(self, training=True):
        if training:
            self.model.train()
            self.loss.train()
            dataloader = self.dataloaders['train']
        else:
            self.model.eval()
            self.loss.eval()
            dataloader = self.dataloaders['test']

        epoch_loss = 0
        correct, total = 0, 0

        if self.load_batch:
            batches = os.listdir(rf".\cache\{self.args.dataset}")
            train_batches = [b for b in batches if "train" in b]
            test_batches = [b for b in batches if "test" in b]
            batches = train_batches if training else test_batches

            for batch in tqdm(batches):
                pools = torch.load(rf".\cache\{self.args.dataset}\{batch}", weights_only=False, map_location=self.device)
                speech_rep = pools['speech_rep']
                eeg_seg = pools['eeg']
                subject_id = pools['subjects']

                loss, correct_b, total_b = self.batch_forward(eeg_seg, speech_rep, subject_id, training)
                correct += correct_b
                total += total_b
                epoch_loss += loss.detach().cpu().item()
        else:
            for data_batch in tqdm(dataloader):
                subject_id, start_stop, speech_rep, eeg_seg = data_batch
                eeg_seg = self.eeg_pca(eeg_seg.float().to(self.device))

                loss, correct_b, total_b = self.batch_forward(eeg_seg, speech_rep, subject_id, training)
                correct += correct_b
                total += total_b
                epoch_loss += loss.detach().cpu().item()

        epoch_loss = epoch_loss / len(dataloader)
        top_k_acc = correct / total * 100 if total != 0 else 0
        if not training and epoch_loss < self.best_loss:
            self.best_loss = epoch_loss
            self.best_epoch = self.epoch
            print(f'New best valid loss {epoch_loss:.3f} at epoch {self.best_epoch}')
            self.best_state = self.copy_state(self.model.state_dict())
            if self.epoch >= 0:
                self.save_dict()
        if not training and top_k_acc > self.best_acc:
            self.best_acc = top_k_acc
            self.best_acc_epoch = self.epoch
            print(f'New best top k acc {top_k_acc:.2f}% at epoch {self.best_acc_epoch}')
            self.best_state = self.copy_state(self.model.state_dict())
            if self.epoch >= 0:
                self.save_dict()
        return epoch_loss, top_k_acc

    def batch_forward(self, eeg_seg, speech_rep, subject_id, training):
        last_dim = eeg_seg.shape[-1] // 200 * 200 + 200
        eeg_seg = F.interpolate(eeg_seg, last_dim)
        speech_rep = F.interpolate(speech_rep, last_dim)
        eeg_seg = eeg_seg.unsqueeze(dim=1)
        subject_id = subject_id.unsqueeze(dim=1)
        # speech_rep = speech_rep.unsqueeze(dim=1)

        # eeg_rep = self.model(eeg_seg, subject_id.to(self.device))
        eeg_rep = self.model(eeg_seg, subject_id.to(self.device))

        if training:
            if self.n_negatives is not None:
                if self.negative_pool is None:
                    self.negative_pool = speech_rep
                    candidate = speech_rep
                else:
                    kept = torch.randperm(self.negative_pool.shape[0])
                    self.negative_pool = self.negative_pool[kept]
                    self.negative_pool = self.negative_pool[:self.n_negatives, ...].clone()
                    candidate = torch.cat((speech_rep, self.negative_pool), dim=0)
                    self.negative_pool = torch.cat((speech_rep, self.negative_pool), dim=0)
            else:
                candidate = speech_rep
            loss = self.loss(eeg_rep, candidate.float().to(self.device))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                loss = self.loss(eeg_rep, speech_rep.float().to(self.device))
            return loss, 0, 0
        else:
            scores = self.loss.get_scores(eeg_rep, speech_rep.float().to(self.device))
            loss = self.loss.get_ce_loss(scores)
            try:
                top_k_preds = scores.topk(int(eeg_rep.shape[0] / self.k), dim=1).indices
                ground_truth = torch.arange(scores.size(0), device=self.device).view(-1, 1)
                correct = (top_k_preds == ground_truth).sum().item()
                total = scores.size(0)
            except RuntimeError:
                pass
            return loss, correct, total
    
    def subject_pred(self, loading=False):
        # if loading:
        #     if self.ckpt is not None:
        #         self.best_state = torch.load(self.ckpt, map_location=self.device)
        #         self.model.load_state_dict(self.best_state)
        #         print(f"Loading ckpt from {self.ckpt}")

        self.model.eval()
        self.loss.eval()
        dataloader = self.dataloaders['test']

        subject_correct = defaultdict(int)
        subject_total = defaultdict(int) 

        for data_batch in dataloader:
            subject_id, start_stop, speech_rep, eeg_seg = data_batch
            eeg_seg = self.eeg_pca(eeg_seg.float().to(self.device))
            eeg_rep = self.model(eeg_seg, subject_id.to(self.device))
            del eeg_seg
            scores = self.loss.get_scores(eeg_rep, speech_rep.float().to(self.device))
            try:
                top_k_preds = scores.topk(int(eeg_rep.shape[0] / self.k), dim=1).indices
                ground_truth = torch.arange(scores.size(0), device=self.device).view(-1, 1)

                for subj in subject_id.unique():
                    mask = (subject_id == subj)
                    subj_indices = mask.nonzero(as_tuple=True)[0]
                    
                    if subj_indices.numel() > 0:
                        subj_top_k_preds = top_k_preds[subj_indices] 
                        subj_ground_truth = ground_truth[subj_indices]
                        
                        # Compute correct predictions for this subject
                        subject_correct[subj.item()] += (subj_top_k_preds == subj_ground_truth).sum().item()
                        subject_total[subj.item()] += subj_indices.numel()
            except RuntimeError:
                pass
        subject_acc = {subj: subject_correct[subj] / subject_total[subj] * 100 for subj in subject_correct}
        return subject_acc

    def eeg_pca(self, x):
        if self.pca is None:
            return x
        self.pca = int(self.pca)
        mean_x = x.mean(dim=1, keepdim=True)
        x_centered = x - mean_x
        x_reshaped = x_centered.permute(0, 2, 1)
        U, S, V = torch.linalg.svd(x_reshaped, full_matrices=False)
        components = V[:, :self.pca, :]
        x_reduced = torch.einsum('bij,bjk->bik', x_reshaped, components.transpose(-1, -2))
        x_reduced = x_reduced.permute(0, 2, 1)
        variance_explained = torch.sum(S[:, :self.pca] ** 2, dim=1) / torch.sum(S ** 2, dim=1)
        return x_reduced

    def save_dict(self):
        if self.save_dict_path is not None:
            if os.path.exists(self.save_dict_path):
                os.remove(self.save_dict_path)
        self.save_dict_path = rf".\ckpt\best_dict_l{self.length}_epoch{self.epoch}.pt"
        os.makedirs(os.path.dirname(self.save_dict_path), exist_ok=True)
        torch.save(self.best_state, f=self.save_dict_path)

    def create_loss(self, loss: str):
        if loss == 'l1':
            return L1Loss()
        elif loss == 'mse':
            return L2Loss()
        elif loss == 'clip':
            loss = ClipLoss()
            if self.optimizer is not None:
                self.optimizer.add_param_group({"params": loss.parameters()})
            return loss
        else:
            raise ValueError(f"Unsupported loss {loss}")

    def plot_process(self):
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.plot(list(range(self.epoch + 1)), self.train_losses, label='Train Loss', color='blue')
        plt.plot(list(range(self.epoch + 1)), self.test_losses, label='Test Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Losses')
        save_path = r".\losses_results\losses.pdf"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', dpi=300)
        plt.close()

        plt.plot(list(range(self.epoch + 1)), self.accs)
        plt.xlabel('Epoch')
        plt.ylabel('Acc (%)')
        plt.title('Top K Accuracy')
        save_path = r".\losses_results\accs.pdf"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', dpi=300)
        plt.close()

    @staticmethod
    def copy_state(state):
        return {k: v.cpu().clone() for k, v in state.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='brennan2019')  # brennan2019  broderick2019
    parser.add_argument('--model', type=str, default='cbramod')  # simplecnn  cbramod  labram
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--max_epoch', type=int, default=300)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--n_negatives', type=int, default=None)
    parser.add_argument('--early_stop_epoch', type=int, default=20)
    parser.add_argument('--eval_every_epcoh', type=int, default=10)
    parser.add_argument('--loss', type=str, default='clip')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--length', default=None)
    parser.add_argument('--split_method', type=str, default='5fold')
    parser.add_argument('--split_fold', type=int, default=0)
    parser.add_argument('--k', type=float, default=5)
    parser.add_argument('--pca', default=None)
    parser.add_argument('--load_batch', action='store_true', default=False)
    parser.add_argument('--load_lbm', action='store_true', default=False)
    parser.add_argument('--foundation_dir', type=str, default=r"E:\NIPS2026\ckpt\cbramod-base.pth")
    args = parser.parse_args()

    if args.datasets == 'brennan2019':
        n_subjects = 32
        out_channels = 240
        num_layers = 1
        from data_process.brennan2019.brennan2019 import *
    elif args.datasets == 'broderick2019':
        n_subjects = 19
        out_channels = 240
        num_layers = 1
        from data_process.broderick2019.broderick2019 import *

    path = rf"datasets\{args.datasets}"
    dataloaders = generate_dataloader(base_path=path, batch_size=args.bs,
                                      split_method=(args.split_method, args.split_fold), length=args.length)
    train_num, test_num = 0, 0
    for data_batch in dataloaders['train']:
        _, _, speech_rep, eeg_seg = data_batch
        if args.pca is None:
            in_channels = eeg_seg[0].shape[0]
        else:
            in_channels = int(args.pca)
        feature_dim = speech_rep[0].shape[0]
        train_num += eeg_seg.shape[0]
    for data_batch in dataloaders['test']:
        _, _, speech_rep, eeg_seg = data_batch
        test_num += eeg_seg.shape[0]
    print(f"Sample numbers {train_num}/{test_num} (train/test)")

    # eeg_model = SimpleConv(in_channels, out_channels, num_layers, feature_dim, n_subjects)
    eeg_model = model_brennan2019.Model(args)

    # eeg_model = BrainMagic(in_channels=in_channels, conv_channels=conv_channels, out_channels=feature_dim,
    #                        n_subjects=n_subjects, num_convblock=num_convblock)

    trainer = Trainer(dataloaders, eeg_model, args)
    trainer.train()