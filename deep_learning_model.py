import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SleepECGCNNLSTM(nn.Module):
    def __init__(self, n_classes=5, hidden_size=128, num_layers=2, dropout=0.2, feat_dim=256, attn_dim=128):
        super().__init__()
        self.cnn2d = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )

        self.proj = nn.Sequential(
            nn.Linear(128 * 4 * 4, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        self.res_proj = nn.Linear(feat_dim, hidden_size*2)

        d_model = hidden_size * 2
        self.mid_query = nn.Linear(d_model, attn_dim, bias=False)
        self.key = nn.Linear(d_model, attn_dim, bias=False)
        self.value = nn.Linear(d_model, attn_dim, bias=False)

        # 输出用 concat(Q_mid, context) 比只用 context 更稳
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(attn_dim + d_model, n_classes)
        )

    def forward(self, x):
        # x: (B, S, F, T)
        B, S, Freq, Time = x.shape
        x = x.view(B * S, 1, Freq, Time)
        feat = self.cnn2d(x)
        feat = self.proj(feat)
        feat = feat.view(B, S, -1)

        lstm_out, _ = self.lstm(feat)    # (B, S, d_model)
        lstm_out = lstm_out + self.res_proj(feat)
        mid = S // 2
        q = self.mid_query(lstm_out[:, mid, :]).unsqueeze(1)  # (B, 1, attn_dim)
        k = self.key(lstm_out)                                  # (B, S, attn_dim)
        v = self.value(lstm_out)                                # (B, S, attn_dim)

        # scaled dot-product attention
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(k.size(-1))  # (B, 1, S)
        attn = F.softmax(scores, dim=-1)                                      # (B, 1, S)
        context = torch.matmul(attn, v).squeeze(1)                            # (B, attn_dim)

        out = torch.cat([lstm_out[:, mid, :], context], dim=-1)               # (B, d_model+attn_dim)
        logits = self.classifier(out)
        return logits
