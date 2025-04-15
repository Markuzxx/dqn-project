import gymnasium as gym
import flappy_bird_gymnasium

from utils import Any


class Environment:
    def __init__(self,
                 env_id: str,
                 render: bool = False,
                 env_params: dict = {}) -> None:
        
        # Create the selected Gymnasium environment
        self.env = gym.make(

            id=env_id,
            render_mode="human" if render else None,
            **env_params

        )

        # Get the environment spaces
        self.action_space       = self.env.action_space
        self.observation_space  = self.env.observation_space

        # Get the environment dimensions
        self.state_dim          = self.observation_space.shape[0] if hasattr(self.observation_space, 'shape') else self.observation_space.n
        self.action_dim         = self.action_space.n if hasattr(self.action_space, 'n') else self.action_space.shape[0]

    def reset(self) -> tuple:

        # Reset the environment and get the initial state

        return self.env.reset()

    def step(self,
             action: Any) -> tuple:
        
        # Take a step in the environment with the given action

        return self.env.step(action)
    
    def close(self) -> None:

        # Close the environment

        self.env.close()
