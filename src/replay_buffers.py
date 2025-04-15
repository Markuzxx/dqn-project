import numpy as np


class ReplayBuffer:

    def __init__(self,
                 capacity: int,
                 state_shape: tuple,
                 dtype=np.float32):
        
        self.capacity       = capacity
        self.position       = 0         # Current index for inserting new experience
        self.size           = 0         # Current size of the buffer
        
        # Preallocate memory for efficiency
        self.states         = np.zeros((capacity, state_shape), dtype=dtype)
        self.actions        = np.zeros(capacity, dtype=np.int32)
        self.rewards        = np.zeros(capacity, dtype=dtype)
        self.next_states    = np.zeros((capacity, state_shape), dtype=dtype)
        self.terminations   = np.zeros(capacity, dtype=np.bool_)

    def push(self,
             state: np.ndarray,
             action: np.ndarray,
             reward: float,
             next_state: np.ndarray,
             terminated: bool) -> None:
        
        # Store the experience in the buffer
        # Overwrite the old experience if the buffer is full
        
        self.states[self.position]          = state
        self.actions[self.position]         = action
        self.rewards[self.position]         = reward
        self.next_states[self.position]     = next_state
        self.terminations[self.position]    = terminated
        
        # Move to the next position, overwriting old experiences when full
        self.position   = (self.position + 1) % self.capacity
        self.size       = min(self.size + 1, self.capacity)
    
    def sample(self,
               batch_size: int) -> list:
        
        # Sample a batch of experiences from the buffer
        
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return [(self.states[i], self.actions[i], self.rewards[i], self.next_states[i], self.terminations[i]) for i in indices]
    
    def get_data(self):

        # Get the data in a dictionary format for easy access

        return {

            'states':       self.states[:self.size],
            'actions':      self.actions[:self.size],
            'rewards':      self.rewards[:self.size],
            'next_states':  self.next_states[:self.size],
            'terminations': self.terminations[:self.size]

        }
    
    def load_data(self, data):

        # Load data from a dictionary format into the buffer

        self.size                       = min(len(data['states']), self.capacity)
        self.states[:self.size]         = data['states'][:self.size]
        self.actions[:self.size]        = data['actions'][:self.size]
        self.rewards[:self.size]        = data['rewards'][:self.size]
        self.next_states[:self.size]    = data['next_states'][:self.size]
        self.terminations[:self.size]   = data['terminations'][:self.size]
        self.position                   = self.size % self.capacity
    
    def __len__(self) -> int:

        # Return the current size of the buffer
        # This allows the buffer to be used with len() function if needed

        return self.size
