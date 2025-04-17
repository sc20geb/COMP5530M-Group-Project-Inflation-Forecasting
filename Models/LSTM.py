import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3, output_size=12, bidirectional=False):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.norm = nn.LayerNorm(hidden_size * self.num_directions)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        # LSTM Forward
        out, _ = self.lstm(x)  # [batch_size, seq_len, hidden*directions]
        out = out[:, -1, :]    # Take output from last time step

        # Post-LSTM Processing
        out = self.norm(out)
        out = self.dropout(out)
        out = self.fc(out)     # Output size: [batch_size, output_size]
        return out
