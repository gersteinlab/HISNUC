
import os
import pickle
import numpy as np
from pathlib import Path
import tensorflow as tf
from logger_config import logger

# Suppress TensorFlow's low-level warnings
tf.get_logger().setLevel('ERROR')


def save_model(
    model: tf.keras.Model,
    history: tf.keras.callbacks.History,
    current_path: str,
    target_dir: str,
    model_name: str,
    y_pred: np.ndarray,
    test_label: np.ndarray
):
    """
    Save trained model, training history, and prediction results to disk.

    Parameters:
    - model (tf.keras.Model): The trained Keras model.
    - history (History): The training history object returned by model.fit().
    - current_path (str): Base directory where files will be saved.
    - target_dir (str): Subdirectory under current_path to save files.
    - model_name (str): File name for the saved model (must end in '.h5').
    - y_pred (np.ndarray): Predictions made on the test set.
    - test_label (np.ndarray): True labels for the test set.
    """

    # Create directory if it doesn't exist
    target_dir_path = Path(current_path) / target_dir
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Ensure proper model file extension
    assert model_name.endswith(".h5"), "model_name should end with '.h5'"
    model_save_path = target_dir_path / model_name

    # Save model
    logger.info("[INFO] Saving model to: %s", model_save_path)
    model.save(model_save_path)

    # Save training history as pickle
    with open(target_dir_path / 'basic_history.pickle', 'wb') as f:
        pickle.dump(history.history, f)

    # Save predictions and ground truth
    np.save(target_dir_path / "y_pred.npy", y_pred)
    np.save(target_dir_path / "test_label.npy", test_label)
