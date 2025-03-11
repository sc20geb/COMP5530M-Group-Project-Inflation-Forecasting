import torch.nn as nn

# N-BEATS Model with Residual Learning
class ResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ResidualBlock, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, output_size)
        )
        self.residual = nn.Linear(input_size, output_size)  # Residual connection

    def forward(self, x):
        x = x.view(x.shape[0], -1)  
        return self.hidden(x) + self.residual(x)  # Residual Learning

class NBeats(nn.Module):
    def __init__(self, input_size, hidden_size=512, output_size=1):
        super(NBeats, self).__init__()
        self.trend_block = ResidualBlock(input_size, hidden_size, output_size)
        self.seasonality_block = ResidualBlock(input_size, hidden_size, output_size)
        self.skip = nn.Linear(input_size, output_size)  # Skip connection

    def forward(self, x):
        trend = self.trend_block(x)
        seasonality = self.seasonality_block(x)
        skip_connection = self.skip(x.view(x.shape[0], -1))  # Shortcut
        return trend + seasonality + skip_connection