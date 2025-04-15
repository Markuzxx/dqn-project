from logit import Logger

import time
import argparse
import itertools

from environment import Environment
from agent import Agent
from utils import *


def train(resume: bool,
          hyperparameters_set: str | None = None,
          session_path: str | None = None) -> None:
    
    '''
    Train a DQN model.
    '''

    if not resume:

        hyperparameters = load_yaml(os.path.join(CONFIG_DIR, 'hyperparameters.yml'))
        
        if hyperparameters_set not in hyperparameters:

            raise ValueError(f"Hyperparameters set '{hyperparameters_set}' not found in configuration file.")

        hyperparameters = hyperparameters[hyperparameters_set]

        num_sessions = count_folders_with_prefix(SESSIONS_DIR, f'{hyperparameters_set}_')
        session_path = os.path.join(SESSIONS_DIR, f'{hyperparameters_set}_{num_sessions}')

        create_session(session_path, hyperparameters)

    else:

        if not os.path.exists(session_path):

            raise FileNotFoundError(f"Session path '{session_path}' does not exist.")

        hyperparameters = load_yaml(os.path.join(session_path, 'data', 'hyperparameters.yml'))

    training_data_path  = os.path.join(session_path, 'data', 'training_data.csv')
    best_model_path     = os.path.join(session_path, 'models', 'best_model.pth')
    last_model_path     = os.path.join(session_path, 'models', 'last_model.pth')
    graphs_path         = os.path.join(session_path, 'graphs', 'training_graphs.png')   # For future implementation

    # Get max steps and penalty parameters
    max_steps           = hyperparameters['training'].get('max_steps', float('inf'))
    max_step_penalty    = hyperparameters['training'].get('max_step_penalty', 0)
    
    data_manager = TrainingDataManager(

        file_name=training_data_path,
        columns={

            'episode': int,
            'reward': float,
            'steps': int,
            'loss': float,
            'epsilon': float,
            'time': float

        }

    )

    logger = Logger(

        name='Logger',
        level='INFO',
        datefmt=DATE_FORMAT,
        log_file_path=os.path.join(session_path, 'logs', 'training_log.log')

    )

    env = Environment(

        env_id=hyperparameters['environment']['env_id'],
        render=False,
        env_params=hyperparameters['environment'].get('env_config', {})

    )

    agent = Agent(

        env=env,
        hyperparameters=hyperparameters['agent']

    )

    if not resume:

        best_reward = None
        start = 0

        logger.info('Training starting...')

    else:

        agent.load_checkpoint(last_model_path)

        best_reward = max(data_manager.data_dict['reward'])
        start = data_manager.data_dict['episode'][-1] + 1

        logger.info('Resuming training...')
    
    try:

        for episode in itertools.count(start=start):

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
                reward = torch.tensor(reward, dtype=torch.float32, device=device)

                # Store experience in agent's memory
                agent.store(state, action, reward, new_state, terminated)

                loss = agent.update()

                if step_count % hyperparameters['agent']['target_update_freq'] == 0:

                    agent.sync_target_network()
                
                # Increase the step counter
                step_count += 1

                # Move to the next state
                state = new_state

            agent.update_epsilon()

            data_manager.add_data(

                episode=episode,
                reward=episode_reward,
                steps=step_count,
                loss=loss,
                epsilon=agent.epsilon,
                time=time.time() - start_time

            )

            if best_reward is None:

                logger.info(

                    f'Episode {episode}: New best reward {episode_reward:.2f}. Saving model...'

                )

                agent.save_checkpoint(best_model_path)
                best_reward = episode_reward

            elif episode_reward > best_reward:

                if best_reward != 0:

                    logger.info(

                        f'Episode {episode}: New best reward {episode_reward:.2f} '
                        f'({(episode_reward - best_reward) / best_reward * 100:+.1f}%). Saving model...'

                    )

                else:

                    logger.info(f"Episode {episode}: New best reward {episode_reward:.2f}. Saving model...")

                agent.save_checkpoint(best_model_path)
                best_reward = episode_reward

    except KeyboardInterrupt:

        logger.info('Training interrupted by the user. Saving data...')
        logger.blank()

        data_manager.save()
        agent.save_checkpoint(last_model_path)


if __name__ == '__main__':

    parser =argparse.ArgumentParser(description='Train a DQN model')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--new', type=str, metavar='HYPERPARAMETERS_SET', help='Train a new DQN model')
    group.add_argument('--resume', type=str, metavar='SESSION_PATH', help='Resume training of an existing DQN model')

    args = parser.parse_args()

    if args.new:
        
        train(

            resume=False,
            hyperparameters_set=args.new

        )

    elif args.resume:

        train(

            resume=True,
            session_path=args.resume

        )
