
import os
import sys
import logging
import datetime
from pathlib import Path

# Base directory for the experiment (current working directory)
MAIN_DIR = os.getcwd()

# Name of the experiment, used in the folder structure
EXP_NAME = "tile-NIC-CNN-matrix"

def get_run_folder(args):
    """
    Generate a unique folder name for the experiment run based on arguments and timestamp.

    Parameters:
    - args: An object containing experiment hyperparameters like learning_rate, test_size, etc.

    Returns:
    - str: A string used as the run folder name.
    """
    args_str = (
        f"lr{args.learning_rate}-"
        f"test_size{args.test_size}-"
        f"batch_size{args.batch_size}-"
        f"epochs{args.epochs}-"
        f"rows{args.image_count}"
    )
    now_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return f"run_{now_str}_{args_str}"

def create_path(parent_dir, child_dirs):
    """
    Create a nested directory structure under the given parent directory.

    Parameters:
    - parent_dir (Path): The base directory to start from.
    - child_dirs (List[str]): A list of subdirectories to create in order.

    Returns:
    - Path: The full path to the final created directory.
    """
    path = parent_dir
    for child_dir in child_dirs:
        path = path / child_dir
        path.mkdir(exist_ok=True)
    return path

def initialize_experiment(args):
    """
    Set up a new experiment directory under 'data/EXP_NAME/' using args to define subfolder names.

    Parameters:
    - args: An object with necessary attributes for naming (e.g., learning rate, epochs, etc.)

    Returns:
    - Path: The full path to the newly created run folder.
    """
    current_path = create_path(Path(MAIN_DIR), ["data", EXP_NAME, get_run_folder(args)])
    return current_path
