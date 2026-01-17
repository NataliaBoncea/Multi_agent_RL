import torch
import torch.nn as nn



# --- RNN POLICY NETWORK ---
class RNNPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, device):
        super(RNNPolicy, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.rnn = nn.GRUCell(hidden_size, hidden_size) # Recurrent Neural Network
        self.fc2 = nn.Linear(hidden_size, output_dim)
        self.device = device
        
    def forward(self, x, h):
        x = torch.relu(self.fc1(x))
        h_next = self.rnn(x, h)
        out = self.fc2(h_next)
        return out, h_next

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size, device=self.device)