
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

## Requirements

- Python 3.10.11 (recommended)
- pip

## Installation

### Clone the repository

```bash
git clone https://github.com/Markuzxx/dqn-project.git
cd dqn-project
```

### Create a virtual environment (recommended)

It is highly recommended to use a virtual environment to isolate the project dependencies.

- **Windows - cmd**

    ```cmd
    py -3.10.11 -m venv venv
    venv\Scripts\activate.bat
    ```

- **Windows - PowerShell**

    ```powershell
    py -3.10.11 -m venv venv
    .\venv\Scripts\Activate.ps1
    ```

- **Linux/macOS**

    ```bash
    python3.10 -m venv venv
    source venv/bin/activate
    ```

### Install dependencies

1. **PyTorch installation**

    Navigate to `pytorch.org` and scroll down untill you find a menu.
    Select:

    - Stable
    - Your OS
    - Python
    - Your compute platform (if you are on MacOS you should select default)

    Copy the command line generated and execute it.

2. **Install other dependecies**

    ```bash
    pip3 install -r requirements.txt
    ```

## How to Use

### Training

To start a new training session with a specific hyperparameter set:

```bash
python3 train.py --new hyperparameters_set
```

This will create a new folder under `sessions/` named like `hyperparameters_set_0`, `hyperparameters_set_1`, etc.

To resume an existing session, provide the path to the session folder:

```bash
python3 train.py --resume sessions/hyperparameters_set_0
```

**Note:** *`--new` and `--resume` are mutually exclusive and one of them is required.*

### Evaluation

To evaluate a trained model:

```bash
python3 evaluate.py sessions/hyperparameters_set_0
```

You can specify which model to use:

- `--model best` (loads the best-performing model)
- `--model last` (loads the most recent checkpoint — default if omitted)

You can also render the environment during evaluation:

```bash
python3 evaluate.py sessions/hyperparameters_set_0 --model best --render
```

## Current Progress

- [x] Virtual environment configured
- [x] Modular project structure
- [x] Basic neural network and replay buffer
- [x] DQN algorithm implementation (testing phase)
- [x] Logging  
- [x] Evaluation integration and model saving
- [x] Optimization

## Future Improvements

Planned features and extensions include:

- Double DQN
- Dueling DQN
- Prioritized Experience Replay (PER)
- More complex, custom environments
