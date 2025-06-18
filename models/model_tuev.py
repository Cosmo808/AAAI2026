import torch
import torch.nn as nn

from .cbramod import CBraMod


class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
        self.backbone = CBraMod(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8
        )
        if param.use_pretrained_weights:
            map_location = torch.device(f'cuda:{param.cuda}')
            self.backbone.load_state_dict(torch.load(param.foundation_dir, map_location=map_location, weights_only=False))
        self.backbone.proj_out = nn.Sequential()
        # self.avgpooling = nn.AdaptiveAvgPool2d(1)
        # self.attn = nn.MultiheadAttention(200, num_heads=8, dropout=0.1)
        self.classifier = nn.Sequential(
            nn.Linear(16*5*200, 5*200),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(5*200, 200),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(200, param.num_of_classes)
        )

    def forward(self, x):
        bz, ch_num, seq_len, patch_size = x.shape
        feats = self.backbone(x)
        feats = feats.contiguous().view(bz, ch_num*seq_len*200)
        # feats = self.avgpooling(feats).contiguous().view(bz, 200)
        out = self.classifier(feats)
        return out

