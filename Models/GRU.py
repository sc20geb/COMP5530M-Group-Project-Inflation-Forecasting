# Define the GRU Model
import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(hidden_size)  # Batch Normalization
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.batch_norm(out[:, -1, :])  # Apply batch norm after GRU
        out = self.fc(out)  # Fully connected layer
        return out
