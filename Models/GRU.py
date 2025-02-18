# import torch.nn as nn

# class GRUModel(nn.Module):
#     def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
#         super(GRUModel, self).__init__()
#         self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
    
#     def forward(self, x):
#         out, _ = self.gru(x)
#         out = self.fc(out[:, -1, :])  # Take the last time step output
#         return out

# import torch.nn as nn

# class GRUModel(nn.Module):
#     def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
#         super(GRUModel, self).__init__()
        
#         # Define GRU layers with return_sequences=True to stack multiple GRUs
#         self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
#         # Layer normalization for stable training
#         self.layer_norm = nn.LayerNorm(hidden_size)
        
#         # Fully connected layer (output)
#         self.fc = nn.Linear(hidden_size, output_size)

#         # Leaky ReLU activation
#         self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

#     def forward(self, x):
#         out, _ = self.gru(x)  # Forward through GRU
#         out = self.layer_norm(out[:, -1, :])  # Normalize last time step output
#         out = self.leaky_relu(out)  # Apply Leaky ReLU activation
#         out = self.fc(out)  # Fully connected output layer
#         return out



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