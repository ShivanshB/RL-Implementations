import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim, n_hidden, num_actions):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, num_actions)
        )

    def forward(self, x):
        return self.layers(x)