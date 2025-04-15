import torch
import torch.nn as nn


class DQN(nn.Module):

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int) -> None:
        
        super(DQN, self).__init__()

        # Standard fully connected network
        self.network = nn.Sequential(

            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)

        )

    def forward(self,
                state: torch.Tensor) -> torch.Tensor:
        
        # Pass the input through the hidden and compute Q-values

        return self.network(state)
