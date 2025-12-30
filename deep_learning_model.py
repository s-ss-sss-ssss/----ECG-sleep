import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np
import random
from pathlib import Path
class SleepECGCNNLSTM(nn.Module):
    def __init__(self, input_length=6000, n_classes=5, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_length = input_length
        self.n_classes = n_classes

        # CNN feature extractor for each epoch
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=51, stride=6, padding=25),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(8),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=17, stride=2, padding=8),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(dropout),

            nn.AdaptiveAvgPool1d(16),  # -> (B, 128, 16)
            nn.Flatten(start_dim=2)    # -> (B, 128*16)
        )

        # LSTM over sequence of epochs
        self.lstm = nn.LSTM(
            input_size=128 * 16,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        # x: (B, seq_len=5, T=6000)
        B, S, T = x.shape
        x = x.view(B * S, 1, T)            # (B*S, 1, 6000)
        features = self.cnn(x)             # (B*S, 2048)
        features = features.view(B, S, -1) # (B, 5, 2048)
        lstm_out, _ = self.lstm(features)  # (B, 5, hidden_size)
        # Predict the middle epoch (index = 2 when S=5)
        out = self.classifier(lstm_out[:, S//2, :])  # (B, n_classes)
        return out