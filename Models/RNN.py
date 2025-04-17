# import torch
# import torch.nn as nn

# class RNNModel(nn.Module):
#     def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1):
#         super(RNNModel, self).__init__()
#         # Define computation device (GPU if available, otherwise CPU)
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         # Define the RNN layer
#         self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
#         # Define a fully connected layer for the final output
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         # Initialize hidden state with zeros
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

#         # Forward propagate through the RNN
#         out, _ = self.rnn(x, h0)
#         # Take the output from the last time step
#         out = out[:, -1, :]
#         out = self.fc(out)
#         return out


import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1, dropout=0.0):
        super(RNNModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Apply dropout only if num_layers > 1 (PyTorch limitation)
        dropout_rate = dropout if num_layers > 1 else 0.0

        # Define the RNN layer with optional dropout
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)

        # Define a fully connected layer for the final output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros on the correct device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # Forward pass through RNN
        out, _ = self.rnn(x, h0)

        # Use the output from the final time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out
