import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils as utils

from environment import Environment
from replay_buffers import ReplayBuffer
from models import DQN
from utils import device


# Dictionary of available optimizers
optimizers = {

    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "SGD": torch.optim.SGD,
    "RMSprop": torch.optim.RMSprop
    
}


class Agent:
    
    def __init__(self,
                 env: Environment,
                 hyperparameters: dict,
                 evaluate_mode: bool = False) -> None:

        self.device = torch.device(device)
        
        self.env = env

        # General settings
        self.gamma          = hyperparameters['gamma']
        self.epsilon        = hyperparameters['epsilon'] if not evaluate_mode else 0
        self.epsilon_decay  = hyperparameters['epsilon_decay']
        self.epsilon_min    = hyperparameters['epsilon_min']
        self.batch_size     = hyperparameters['batch_size']

        # Memory settings
        self.capacity       = hyperparameters['capacity']

        # Optimizer settings
        self.learning_rate  = hyperparameters['learning_rate']
        self.weight_decay   = hyperparameters.get('weight_decay', 0.0)
        self.optimizer_type = hyperparameters['optimizer']
        
        # Create policy and target networks
        self.policy_network = self._create_neural_network(hyperparameters)
        self.target_network = self._create_neural_network(hyperparameters)
        
        # Memory for storing experiences
        self.memory         = self._create_memory()
        self.push           = self._push    # ready for future updates
        
        # Other functions (ready for future updates)
        self.loss_fn                    = nn.MSELoss()
        self.compute_q_target           = self._compute_q_target
        self.sample_batch               = self._sample_batch
        self.update_epsilon             = self._update_epsilon

        # Optimizer for the policy network
        self.optimizer = optimizers[hyperparameters['optimizer']](

            self.policy_network.parameters(),
            lr=hyperparameters['learning_rate'],
            weight_decay=hyperparameters['weight_decay']

        )

    def _create_neural_network(self,
                               hyperparameters: dict) -> DQN:
        
        # Create the neural network
        
        return DQN(

            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim,
            hidden_dim=hyperparameters['hidden_dim']

        ).to(device)
    
    def _create_memory(self) -> ReplayBuffer:

        # Create memory buffer
        
        return ReplayBuffer(

            capacity=self.capacity,
            state_shape=self.env.state_dim

        )
    
    def _push(self,
              transition) -> None:

        # Push the transition to memory

        self.memory.push(*transition)

    def _sample_batch(self) -> list:

        # Sample a mini-batch from memory

        batch = self.memory.sample(self.batch_size)

        return batch
    
    def _compute_q_target(self,
                          next_states: torch.Tensor,
                          rewards: torch.Tensor,
                          terminations: torch.Tensor) -> torch.Tensor:
        
        # Compute the Q-target using the target network (for standard DQN)
        
        with torch.no_grad():

            next_q = self.target_network(next_states).max(dim=1)[0]
            return rewards + self.gamma * next_q * (1 - terminations)
    
    def _update_epsilon(self) -> None:
        
        # Decay epsilon for exploration-exploitation balance

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
    
    def act(self,
            state: torch.Tensor) -> torch.Tensor:
        
        # Select an action (random action or best action)
        
        # Select an action based on epsilon-greedy policy
        if np.random.rand() < self.epsilon:

            action = self.env.action_space.sample() # Random action for exploration
            return torch.tensor(action, dtype=torch.int64, device=device)

        # Use the policy network to choose the best action
        with torch.no_grad():

            return self.policy_network(state.unsqueeze(dim=0)).squeeze().argmax()
        
    def store(self,
              *transition) -> None:
        
        # Store the transition in memory

        self.push(transition)

    def update(self) -> float:

        # Update the policy network using a mini-batch of experiences

        # Check if there are enough experiences in memory
        if len(self.memory) < self.batch_size: return 0.0

        # Sample a batch of experiences (and priorities if enabled)
        batch = self.sample_batch()

        # Convert the batch to tensors
        states, actions, rewards, next_states, terminations = zip(*batch)

        # Stack tensors to create batch tensors
        states          = torch.as_tensor(np.array(states), dtype=torch.float32, device=device)
        actions         = torch.as_tensor(np.array(actions), dtype=torch.int64, device=device)
        rewards         = torch.as_tensor(np.array(rewards), dtype=torch.float32, device=device)
        next_states     = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=device)
        terminations    = torch.as_tensor(np.array(terminations), dtype=torch.float32, device=device)

        # Compute Q-values for current states and actions
        current_q = self.policy_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute the target Q-values
        target_q = self.compute_q_target(next_states, rewards, terminations)
        
        # Compute the loss
        loss = self.loss_fn(current_q, target_q)

        # Perform a step of gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def sync_target_network(self,
                            tau: float = 1.0) -> None:
        
        # Target network synchronization

        # Convert all parameters of the networks to a single vector each
        policy_vector = utils.parameters_to_vector(self.policy_network.parameters())
        target_vector = utils.parameters_to_vector(self.target_network.parameters())
        
        # Perform linear interpolation between target and policy parameters.
        new_target_vector = target_vector.lerp(policy_vector, tau)
        
        # Update the target network parameters with the interpolated vector
        utils.vector_to_parameters(new_target_vector, self.target_network.parameters())

    def save_checkpoint(self,
                        file_path: str) -> None:
        
        # Save the model and optimizer state to a file
        
        checkpoint = {

            'model_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'memory': self.memory.get_data()

        }

        torch.save(checkpoint, file_path)

    def load_checkpoint(self,
                        file_path: str) -> None:
        
        # Load the model and optimizer state from a file

        checkpoint = torch.load(file_path, map_location=self.device, weights_only=False)
        self.policy_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.memory.load_data(checkpoint['memory'])

    def evaluate(self) -> None:

        # Set the agent to evaluation mode (no exploration)

        self.epsilon = 0.0
