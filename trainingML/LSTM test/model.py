# lstm_model.py
import torch
import torch.nn as nn

class LSTMRegressor(nn.Module):
    def __init__(self, num_features, hidden_size=64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 3)   # t+1, t+5, t+10
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]   # last timestep
        return self.fc(out)