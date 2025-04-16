
# dqn-project

A Deep Q-Learning (DQN) agent implementation for decision-making in a simulated environment. The agent leverages two neural networks (policy and target) to approximate the Q-function and learn optimal policies based on reward signals. Exploration is managed to balance new experience acquisition and exploitation of known behaviors. This project serves as a solid foundation, ready for future upgrades with more advanced reinforcement learning techniques.

## Motivation

This project was developed to gain hands-on experience with the fundamental mechanisms of reinforcement learning. It combines programming, mathematics, and artificial intelligence concepts to create an adaptive agent capable of improving its behavior through interaction with an environment.

## Project Structure

- `config/`: contains configuration files (e.g., `hyperparameters.yml`)
- `sessions/`: saved training sessions and models
- `agent.py`: DQN agent implementation
- `environment.py`: simulated environment (customizable)
- `evaluate.py`: agent evaluation script
- `models.py`: neural network architecture
- `replay_buffer.py`: experience replay memory
- `train.py`: training script
- `utils.py`: helper functions, global constants, and small classes

## Setup Instructions

1. **Clone the repository**

    ```powershell
    git clone https://github.com/Markuzxx/dqn-project.git
    cd dqn-project
    ```

2. **Create and activate a virtual environment**

    On Windows (using cmd)

    ```cmd
    python -m venv venv
    venv\Scripts\activate.bat
    ```

    On Windows (using PowerShell)

    ```powershell
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    ```

3. **Install dependencies**

    ```powershell
    pip install -r requirements.txt
    ```

## How to Use

### Training

To start a new training session with a specific hyperparameter set:

```powershell
python train --new hyperparameters_set
```

This will create a new folder under `sessions/` named like `hyperparameters_set_0`, `hyperparameters_set_1`, etc.

To resume an existing session, provide the path to the session folder:

```powershell
python train --resume sessions/hyperparameters_set_0
```

**Note:** *`--new` and `--resume` are mutually exclusive and one of them is required.*

### Evaluation

To evaluate a trained model:

```powershell
python evaluate sessions/hyperparameters_set_0
```

You can specify which model to use:

- `--model best` (loads the best-performing model)
- `--model last` (loads the most recent checkpoint — default if omitted)

You can also render the environment during evaluation:

```powershell
python evaluate sessions/hyperparameters_set_0 --model best --render
```

## Current Progress

- [x] Virtual environment configured
- [x] Modular project structure
- [x] Basic neural network and replay buffer
- [x] DQN algorithm implementation (testing phase)
- [ ] Evaluation integration and model saving
- [ ] Optimization and logging

## Future Improvements

Planned features and extensions include:

- Double DQN
- Dueling DQN
- Prioritized Experience Replay (PER)
- More complex, custom environments
