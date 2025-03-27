import torch
import torch.nn as nn

# N-BEATSx Model for Time Series Forecasting
# ------------------------------------------
# This model is an extension of N-BEATS that incorporates exogenous variables.
# It consists of multiple fully connected blocks that learn trends and seasonality.
# Define N-BEATSx Model
class NBeatsx(nn.Module):
    def __init__(self, input_size, exog_size, num_blocks=8, theta_size=1):
        '''
        Implements the N-BEATSx architecture for time series forecasting.

        This model consists of:
        - Multiple fully connected blocks (default: 8 blocks).
        - Exogenous variables incorporated into the input.
        - Batch normalization and dropout for better generalization.

        Parameters:
        -----------
        input_size: int
            The number of input features (time-series data length).
        exog_size: int
            The number of exogenous (external) features.
        num_blocks: int, default=8
            The number of fully connected blocks in the model.
        theta_size: int, default=1
            The number of output neurons (typically 1 for forecasting).

        Returns:
        --------
        Output tensor containing the predicted time series values.
        '''
        super().__init__()

        # Create multiple fully connected blocks
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size + exog_size, 1024),  # First fully connected layer with increased neurons
                nn.BatchNorm1d(1024),  # Batch normalization for stable training
                nn.ReLU(),  # Activation function
                nn.Dropout(0.3),  # Dropout to prevent overfitting
                nn.Linear(1024, 512),  # Second fully connected layer
                nn.ReLU(),  # Activation function
                nn.Linear(512, theta_size)   # Final output layer
            ) for _ in range(num_blocks)  # Repeat for the specified number of blocks
        ])

    def forward(self, x, exog):
        '''
        Forward pass of the N-BEATSx model.

        The input consists of:
        - Time-series features (x)
        - Exogenous features (exog)

        Each block processes the combined input, and the outputs are summed
        to form the final forecast.

        Parameters:
        -----------
        x: torch.Tensor
            Input tensor of shape (batch_size, sequence_length).
        exog: torch.Tensor
            Exogenous variable tensor of shape (batch_size, exog_size).

        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, 1) containing forecasted values.
        '''
        forecast = torch.zeros_like(x[:, -1:], dtype=torch.float32)
        for block in self.blocks:
            forecast += block(torch.cat((x, exog), dim=1))  # Concatenate time-series and exogenous variables
        return forecast
