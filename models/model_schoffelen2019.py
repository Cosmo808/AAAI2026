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
        # down_size = 128
        # self.downsample = nn.Sequential(
        #     nn.Conv1d(273, down_size, kernel_size=3, stride=1, padding=1),
        #     nn.GroupNorm(num_groups=16, num_channels=down_size),
        # )

        self.model_name = args.model
        if args.model == 'simplecnn':
            self.backbone = SimpleConv(
                in_channels=273, out_channels=480, num_layers=1,
                feature_dim=1024, n_subjects=30
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
                    self.backbone.load_state_dict(torch.load(args.foundation_dir, map_location=map_location, weights_only=False))
                self.backbone.proj_out = nn.Identity()
            elif args.model == 'labram':
                self.backbone = generate_labram()

            self.subject_layer = SubjectLayers(273, 273, 30)

            self.final = nn.Sequential(
                nn.Conv1d(273, 480, kernel_size=1, stride=1),
                # nn.BatchNorm1d(2 * 128),
                nn.GroupNorm(num_groups=16, num_channels=480),
                nn.Dropout(0.5),
                nn.GELU(),
                nn.ConvTranspose1d(480, 1024, kernel_size=1, stride=1),
            )

    def forward(self, x, subjects=None):
        B, L, C, T = x.shape   # [bs, 5, 273, 5 * 120 = 600]
        x = rearrange(x, 'B L C T -> (B L) C T')
        # x = self.downsample(x)

        if self.model_name == 'simplecnn':
            if subjects is None:
                subjects = torch.zeros(B * L).to(torch.int64).to(x.device)
            x = self.backbone(x, subjects)
            return x

        else:
            if subjects is not None:
                subjects = rearrange(subjects, 'B L -> (B L)')
                x = self.subject_layer(x, subjects)
            x = rearrange(x, 'BL C (a t) -> BL C a t', t=200)
            x = self.backbone(x)
            x = rearrange(x, 'BL C a t -> BL C (a t)', t=200)
            x = self.final(x)
            return x

