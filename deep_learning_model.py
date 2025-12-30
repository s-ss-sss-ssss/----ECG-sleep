import torch
import torch.nn as nn

class SleepECGCNNLSTM(nn.Module):
    def __init__(self, n_classes=5, hidden_size=128, num_layers=2, dropout=0.2, feat_dim=256):
        super().__init__()

        # 2D CNN feature extractor for each epoch's spectrogram
        # input per epoch: (1, F, T)
        self.cnn2d = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),          # (F/2, T/2)
            nn.Dropout(dropout),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),          # (F/4, T/4)
            nn.Dropout(dropout),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Make it size-agnostic: output fixed (128, 4, 4)
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),                          # -> (B*S, 128*4*4=2048)
        )

        # Project CNN features to a compact embedding for LSTM
        self.proj = nn.Sequential(
            nn.Linear(128 * 4 * 4, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # LSTM over sequence of epochs (context window)
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, n_classes)
        )

    def forward(self, x):
        """
        x: (B, S, F, T)  from STFT dataset
        """
        B, S, F, T = x.shape

        # merge batch and sequence, add channel dim
        x = x.view(B * S, 1, F, T)          # (B*S, 1, F, T)
        feat = self.cnn2d(x)                # (B*S, 2048)
        feat = self.proj(feat)              # (B*S, feat_dim)

        feat = feat.view(B, S, -1).mean(dim=1)         # (B, S, feat_dim)
        # lstm_out, _ = self.lstm(feat)       # (B, S, 2*hidden)

        # out = self.classifier(lstm_out[:, S // 2, :])  # predict center epoch
        out = self.classifier(feat)
        return out
