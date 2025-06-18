import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
from models.cbramod import CBraMod


class Model(nn.Module):
    def __init__(self, args: Any):
        super(Model, self).__init__()
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8
        )
        if args.load_lbm:
            map_location = torch.device(f'cuda:0')
            self.backbone.load_state_dict(torch.load(args.foundation_dir, map_location=map_location, weights_only=False))
        self.backbone.proj_out = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(20*3*200, 5*200),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(5*200, 200),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=200, nhead=4, dim_feedforward=2048, batch_first=True, activation=F.gelu, norm_first=True
        )
        self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1, enable_nested_tensor=False)
        self.classifier = nn.Linear(200, 1)

    def forward(self, x):
        bz, seq_len, ch_num, patch_size = x.shape  # [bs, 5, 20, 600]
        x = x.contiguous().view(bz * seq_len, ch_num, 3, 200)
        epoch_features = self.backbone(x)
        epoch_features = epoch_features.contiguous().view(bz, seq_len, ch_num * 3 * 200)
        epoch_features = self.head(epoch_features)
        seq_features = self.sequence_encoder(epoch_features)
        out = self.classifier(seq_features)
        return out

