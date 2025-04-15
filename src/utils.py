import torch

import yaml

import os
import csv


# Use GPU (CUDA cores) if available, otherwise fallback to CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

DATE_FORMAT     = '%Y-%m-%d %H:%M:%S'
CONFIG_DIR      = 'config'
SESSIONS_DIR    = 'sessions'

Any             = object


def count_folders_with_prefix(directory: str,
                              prefix: str) -> int:
    
    # Count the number of folders in a directory that start with a given prefix.
    
    if not os.path.exists(directory): return 0
    
    return sum(1 for name in os.listdir(directory) if name.startswith(prefix))


def save_yaml(data: dict,
              file_path: str) -> None:
    
    # Save a dictionary to a YAML file.

    with open(file_path, 'w') as file:

        yaml.dump(data, file, default_flow_style=False)

def load_yaml(file_path: str) -> dict:

    # Load a dictionary from a YAML file.

    with open(file_path, 'r') as file:

        content = yaml.safe_load(file)

    return content


def create_session(session_path: str,
                   hyperparameters: dict) -> None:
    
    # Create a session folder with sub-folders for data, logs, models, and graphs.

    # Make session folder
    os.makedirs(os.path.join(session_path))

    # Make all session sub-folders
    os.makedirs(os.path.join(session_path, 'data'))
    os.makedirs(os.path.join(session_path, 'logs'))
    os.makedirs(os.path.join(session_path, 'models'))
    os.makedirs(os.path.join(session_path, 'graphs'))

    # Save hyperparameters
    save_yaml(hyperparameters, os.path.join(session_path, 'data', 'hyperparameters.yml'))


class TrainingDataManager:

    # A class to manage training data for a reinforcement learning agent.

    def __init__(self,
                 file_name: str,
                 columns: dict) -> None:
        
        self.file_name = file_name
        
        self.columns = columns
        self.data_dict = {key: [] for key in columns}
        
        self.load()

    def add_data(self,
                 **data) -> None:
        
        # Add data to the training data manager.
        
        for key, value in data.items():

            self.data_dict[key].append(value)

    def save(self) -> None:

        # Save the training data to a CSV file.
        
        keys = self.data_dict.keys()
        rows = zip(*self.data_dict.values())
        
        with open(self.file_name, mode='w', newline='') as file:

            writer = csv.writer(file)
            writer.writerow(keys)
            writer.writerows(rows)

    def load(self) -> None:

        # Load the training data from a CSV file.
        
        try:

            with open(self.file_name, mode='r', newline='') as file:

                reader = csv.DictReader(file)

                for row in reader:

                    for key in self.data_dict:
                        
                        self.data_dict[key].append(self.columns[key](row[key]))

        except FileNotFoundError: pass
