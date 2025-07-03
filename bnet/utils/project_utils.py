import torch
import numpy as np
import random
import yaml
import os

def set_seed(seed):
    """
    Sets the random seed for reproducibility.

    Args:
        seed (int): The seed to use.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path):
    """
    Loads a YAML configuration file.

    Args:
        config_path (str): The path to the YAML config file.

    Returns:
        dict: The configuration as a dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_dirs(paths):
    """
    Creates directories if they do not already exist.

    Args:
        paths (list of str): A list of directory paths to create.
    """
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)