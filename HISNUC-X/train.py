"""
Main training script for gene expression prediction from histopathology image patches,
nucleus features, and demographic metadata.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from logger_config import setup_logger, logger
import experiment_setup
import data_setup, processingslide
import model_builder, save_models, plots

# -------------------- Argument Parser --------------------
parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('--target_genes', type=int, default=100)
parser.add_argument('--seed', type=int, default=20)
parser.add_argument('--test_size', type=float, default=0.33)
parser.add_argument('--image_count', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=0.00004)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--csv_file', type=str, required=True)
parser.add_argument('--image_dir', type=str, required=True)
parser.add_argument('--qupath_feature_path', type=str, required=True)
parser.add_argument('--metadata_file', type=str, required=True)
parser.add_argument('--tissue_name', type=str, required=True)
parser.add_argument('--tau_threshold', type=float, default=0.85)
parser.add_argument('--expression_threshold', type=float, default=0.0)
parser.add_argument('--demographic_features', type=int, default=25)
args = parser.parse_args()

# -------------------- Experiment Setup --------------------
current_path = experiment_setup.initialize_experiment(args)
setup_logger(current_path)
logger.info(f"TensorFlow version: {tf.__version__}")
logger.info(tf.config.list_physical_devices('GPU'))
logger.info('Experiment setup complete.\n')

# -------------------- Data Preparation --------------------
train_data, nucleus_train, train_labels, val_data, nucleus_val, val_labels, test_data, nucleus_test, test_labels = data_setup.create_datasets(
    args.csv_file,
    args.image_dir,
    args.qupath_feature_path,
    args.metadata_file,
    args.tissue_name,
    args.tau_threshold,
    args.expression_threshold,
    args.target_genes,
    args.seed,
    args.test_size
)

logger.info("Processing image patches and stacking top 8 patches per slide...")
train_image, train_nucleus, train_labels, Image_ID_train, _ = processingslide.stack_data(*processingslide.process_all_slides(train_data, nucleus_train, train_labels, True, args.image_dir))
val_image, val_nucleus, val_labels, Image_ID_val, _ = processingslide.stack_data(*processingslide.process_all_slides(val_data, nucleus_val, val_labels, False, args.image_dir))
test_image, test_nucleus, test_labels, Image_ID_test, _ = processingslide.stack_data(*processingslide.process_all_slides(test_data, nucleus_test, test_labels, False, args.image_dir))

# -------------------- Feature Splitting --------------------
def split_features(nucleus_data):
    return nucleus_data[:, :args.demographic_features], nucleus_data[:, args.demographic_features:]

train_demographic, train_nucleus_only = split_features(train_nucleus)
val_demographic, val_nucleus_only = split_features(val_nucleus)
test_demographic, test_nucleus_only = split_features(test_nucleus)

# -------------------- Model Building --------------------
input_shape = train_image.shape[1:]
output_units = train_labels.shape[1]
nucleus_input = train_nucleus.shape[1]
nucleus_feature_input_shape = nucleus_input - args.demographic_features
model = model_builder.build_model(output_units, input_shape, args.demographic_features, nucleus_feature_input_shape)
logger.info("Model Summary:")
model.summary(print_fn=logger.info)

# -------------------- Compile and Train --------------------
os.chdir(current_path)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    args.learning_rate, decay_steps=10000, decay_rate=0.95, staircase=True
)
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule), loss="mse")

callbacks = [
    ModelCheckpoint(filepath="best_model.keras", save_best_only=True, monitor="val_loss"),
    EarlyStopping(monitor='val_loss', patience=30)
]

history = model.fit(
    [train_image, train_demographic, train_nucleus_only], train_labels,
    validation_data=([val_image, val_demographic, val_nucleus_only], val_labels),
    epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks
)

# -------------------- Evaluation and Saving --------------------
y_pred = model.predict([test_image, test_demographic, test_nucleus_only])
results = model.evaluate([test_image, test_demographic, test_nucleus_only], test_labels)
logger.info(f"Test MSE: {results:.4f}")

save_models.save_model(
    model=model,
    history=history,
    current_path=current_path,
    target_dir="models",
    model_name="basic.h5",
    y_pred=y_pred,
    test_label=test_labels
)

# -------------------- Plotting --------------------
logger.info("Generating plots...")
plots.all_plots(history, y_pred, test_labels, Image_ID_test, current_path)
logger.info("Done.")
