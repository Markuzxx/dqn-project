from logit import Logger

import time
import argparse

from environment import Environment
from agent import Agent
from utils import *


def evaluate(render: bool,
             session_path: str,
             selected_model: str) -> None:
    
    '''
    Evaluate a trained DQN model.
    '''

    if not os.path.exists(session_path):
        
        raise FileNotFoundError(f"Session path '{session_path}' does not exist.")

    hyperparameters = load_yaml(os.path.join(session_path, 'data', 'hyperparameters.yml'))

    data_path           = os.path.join(session_path, 'data', 'evaluation_data.csv')
    best_model_path     = os.path.join(session_path, 'models', 'best_model.pth')
    last_model_path     = os.path.join(session_path, 'models', 'last_model.pth')
    graphs_path         = os.path.join(session_path, 'graphs', 'evaluation_graphs.png')
    selected_model_path = best_model_path if selected_model == 'best' else last_model_path

    # Get max steps and penalty parameters
    max_steps = hyperparameters['training'].get('max_steps', float('inf'))
    max_step_penalty = hyperparameters['training'].get('max_step_penalty', 0)

    logger = Logger(

        name='Logger',
        level='INFO',
        datefmt=DATE_FORMAT,
        log_file_path=os.path.join(session_path, 'logs', 'evaluation_log.log')

    )

    env = Environment(

        env_id=hyperparameters['environment']['env_id'],
        render=render,
        env_params=hyperparameters['environment'].get('env_config', {})

    )

    agent = Agent(

        env=env,
        hyperparameters=hyperparameters['agent']

    )

    agent.load_checkpoint(selected_model_path)
    agent.evaluate()

    last_reward = None

    logger.info('Evaluation starting...')
    
    try:

        for episode in range(10):

            start_time = time.time()
            # Reset environment for the new episode
            state, _ = env.reset()
            # Convert state to tensor
            state = torch.tensor(state, dtype=torch.float32, device=device)

            terminated = False
            step_count = 0
            episode_reward = 0.0

            while not terminated and step_count < max_steps:

                action = agent.act(state)

                # Step the environment with the selected action
                new_state, reward, terminated, _, _ = env.step(action.item())
                
                # Accumulate reward and penalize if max steps are exceeded
                episode_reward += reward + int(step_count >= max_steps) * max_step_penalty

                # Convert new state, reward and terminated to tensors
                new_state = torch.tensor(new_state, dtype=torch.float32, device=device)
                
                # Increase the step counter
                step_count += 1

                # Move to the next state
                state = new_state

            if last_reward is not None and last_reward != 0:

                logger.info(

                    f'Episode {episode}: Reward {episode_reward:.2f} '
                    f'({(episode_reward - last_reward) / last_reward * 100:+.1f}%).'

                )
                
            else:

                logger.info(f"Episode {episode}: Reward {episode_reward:.2f}.")

            last_reward = episode_reward

    except KeyboardInterrupt:

        logger.info('Evaluation interrupted by the user.')

    else:

        logger.info('Evaluation terminated.')
    
    logger.blank()


if __name__ == '__main__':
    parser =argparse.ArgumentParser(description='Evaluate a trained DQN model')

    parser.add_argument('session_path', type=str, metavar='SESSION_PATH', help='Path to the session directory.')
    parser.add_argument("--model", choices=["best", "last"], default="last", help="Evaluate either the 'best' or 'last' model (default: 'last').")
    parser.add_argument('--render', action='store_true', help='Render flag.')

    args = parser.parse_args()

    evaluate(

        render=args.render,
        session_path=args.session_path,
        selected_model=args.model

    )
