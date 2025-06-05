import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from models.cbramod import CBraMod
from models.labram import generate_labram
from models.simplecnn import SimpleConv, SubjectLayers
from typing import Optional, Any, Union, Callable


class Model(nn.Module):
    def __init__(self, args: Any):
        super().__init__()

        self.model_name = args.model
        if args.model == 'simplecnn':
            self.backbone = SimpleConv(
                in_channels=60, out_channels=240, num_layers=1,
                feature_dim=768, n_subjects=32
            )
        else:
            if args.model == 'cbramod':
                self.backbone = CBraMod(
                    in_dim=200, out_dim=200, d_model=200,
                    dim_feedforward=800, seq_len=30,
                    n_layer=12, nhead=8
                )
                if args.load_lbm:
                    map_location = torch.device(f'cuda:0')
                    self.backbone.load_state_dict(torch.load(args.foundation_dir, map_location=map_location))
                self.backbone.proj_out = nn.Identity()
            elif args.model == 'labram':
                self.backbone = generate_labram()

            self.subject_layer = SubjectLayers(60, 60, 32)

            self.final = nn.Sequential(
                nn.Conv1d(60, 4 * 60, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.GroupNorm(num_groups=16, num_channels=4 * 60),
                nn.Dropout(0.5),
                nn.GELU(),
                nn.Conv1d(4 * 60, 8 * 60, kernel_size=1, stride=1),
                nn.GroupNorm(num_groups=16, num_channels=8 * 60),
                nn.Dropout(0.5),
                nn.GELU(),
                nn.ConvTranspose1d(8 * 60, 768, kernel_size=1, stride=1),
            )

    def forward(self, x, subjects=None):
        B, L, C, T = x.shape   # [bs, 5, 60, 8.334 * 120 = 1000]

        if self.model_name == 'simplecnn':
            x = rearrange(x, 'B L C T -> (B L) C T')
            if subjects is None:
                subjects = torch.zeros(B * L).to(torch.int64).to(x.device)
            x = self.backbone(x, subjects)
            return x

        else:
            x = rearrange(x, 'B L C T -> (B L) C T')
            if subjects is not None:
                subjects = rearrange(subjects, 'B L -> (B L)')
                x = self.subject_layer(x, subjects)
            x = rearrange(x, 'BL C (a t) -> BL C a t', C=C, t=200)
            x = self.backbone(x)
            x = rearrange(x, 'BL C a t -> BL C (a t)', C=C, t=200)
            x = self.final(x)
            return x

