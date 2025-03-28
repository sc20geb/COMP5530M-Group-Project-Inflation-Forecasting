import torch.nn as nn

# Residual Block for N-BEATS Model
# --------------------------------
# This block applies a fully connected neural network with a residual connection.
# It allows the model to learn from both transformed and original input data.
class ResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        '''
        Implements a Residual Block for the N-BEATS model.
        
        This block contains:
        - Two fully connected layers with LeakyReLU activations.
        - A Batch Normalization layer for stable training.
        - A Residual Connection that helps prevent vanishing gradients.

        Parameters:
        -----------
        input_size: int
            The number of input features.
        hidden_size: int
            The number of neurons in the hidden layers.
        output_size: int
            The number of output features.

        Returns:
        --------
        Output tensor with residual learning applied.
        '''
        super(ResidualBlock, self).__init__()
        
        # Fully Connected Layers with Batch Normalization and LeakyReLU Activation
        self.hidden = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),   # Normalizes activations for stable training
            nn.LeakyReLU(0.1),   # Leaky ReLU to prevent dying neurons
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, output_size)
        )
        # Residual Connection (Shortcut Path)
        self.residual = nn.Linear(input_size, output_size)

    def forward(self, x):
        '''
        Forward pass of the Residual Block.

        The input passes through:
        - A hidden layer transformation.
        - A residual connection that directly connects input to output.

        Parameters:
        -----------
        x: torch.Tensor
            The input tensor of shape (batch_size, input_size).

        Returns:
        --------
        torch.Tensor
            Output tensor of the same shape as input.
        '''
        x = x.view(x.shape[0], -1)    # Flatten input
        return self.hidden(x) + self.residual(x)  # Residual Learning

# N-BEATS Model for Time Series Forecasting
# -----------------------------------------
# This model consists of two main blocks:
# - A Trend Block (captures long-term changes in data).
# - A Seasonality Block (captures recurring patterns in data).
# - A Skip Connection that helps retain original input features.
class NBeats(nn.Module):
    def __init__(self, input_size, hidden_size=512, output_size=1):
        '''
        Implements the N-BEATS architecture for time series forecasting.

        This model contains:
        - A Trend Block: Learns long-term trend information.
        - A Seasonality Block: Captures periodic fluctuations in the data.
        - A Skip Connection: Allows the model to retain raw input information.

        Parameters:
        -----------
        input_size: int
            The number of input features (sequence length).
        hidden_size: int, default=512
            The number of neurons in the hidden layers.
        output_size: int, default=1
            The number of output features (forecasted value).

        Returns:
        --------
        Output tensor containing the predicted time series values.
        '''
        super(NBeats, self).__init__()
        # Trend Block - Learns long-term patterns in the data
        self.trend_block = ResidualBlock(input_size, hidden_size, output_size)

        # Seasonality Block - Captures periodic patterns in the data
        self.seasonality_block = ResidualBlock(input_size, hidden_size, output_size)

        self.skip = nn.Linear(input_size, output_size)  # Skip connection - Allows direct mapping of input to output

    def forward(self, x):
        '''
        Forward pass of the N-BEATS model.

        The input is passed through:
        - A Trend Block to capture long-term variations.
        - A Seasonality Block to capture periodic patterns.
        - A Skip Connection to retain direct input information.

        Parameters:
        -----------
        x: torch.Tensor
            The input tensor of shape (batch_size, input_size).

        Returns:
        --------
        torch.Tensor
            Output tensor containing forecasted values.
        '''
        trend = self.trend_block(x)  # Pass input through the trend block
        seasonality = self.seasonality_block(x)  # Pass input through the seasonality block
        skip_connection = self.skip(x.view(x.shape[0], -1))  # skip connection
        return trend + seasonality + skip_connection