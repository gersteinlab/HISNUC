
import os
import pickle
import numpy as np
from pathlib import Path
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow info/warning logs
from logger_config import logger


def save_model(model: tf.keras.Model,
               history: tf.keras.callbacks.History,
               current_path: str,
               target_dir: str,
               model_name: str,
               y_pred: np.ndarray,
               test_label: np.ndarray):
    """
    Saves the trained Keras model, training history, and predictions to disk.

    Parameters:
    - model: Trained Keras model to be saved.
    - history: Training history object returned by model.fit().
    - current_path: Base path where outputs will be saved.
    - target_dir: Subdirectory name under current_path to store model files.
    - model_name: Filename for the saved model (must end with ".h5").
    - y_pred: Numpy array of model predictions on test data.
    - test_label: Ground truth labels for test data.
    """
    # Create the target directory
    target_dir_path = Path(current_path) / target_dir
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Validate model filename
    assert model_name.endswith(".h5"), "model_name should end with '.h5'"
    model_save_path = target_dir_path / model_name

    # Save the model
    logger.info("[INFO] Saving model to: %s", model_save_path)
    model.save(model_save_path)

    # Save training history
    with open(target_dir_path / 'basic_history.pickle', 'wb') as f:
        pickle.dump(history.history, f)

    # Save predictions and ground truth
    np.save(target_dir_path / "y_pred.npy", y_pred)
    np.save(target_dir_path / "test_label.npy", test_label)
