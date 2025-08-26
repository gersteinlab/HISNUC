
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Project modules
import data_setup, processingslide
import experiment_setup, model_builder, save_models, plots
from logger_config import setup_logger, logger

# Argument parsing
parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('--target', type=int, default=1)
parser.add_argument('--seed', type=int, default=20)
parser.add_argument('--test_size', type=float, default=0.33)
parser.add_argument('--image_count', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=0.00005)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--csv_file', type=str, default='path_to_expression_file')
parser.add_argument('--image_dir', type=str, default="path_to_output_directory")
parser.add_argument('--qupath_feature_path', type=str, default="path_to_qupath_feature.csv")
parser.add_argument('--metadata_file', type=str, default="/path/to/metadata.csv")
parser.add_argument('--nuc_feature', nargs='+', default=[], help="List of nucleus features")
args = parser.parse_args()

# ==== Setup ====
if args.image_count > 800:
    logger.info("Error: image_count cannot be more than 800")
    exit()

current_path = experiment_setup.initialize_experiment(args)
setup_logger(current_path)
logger.info(f"TensorFlow version: {tf.__version__}")
logger.info(f"GPUs: {tf.config.list_physical_devices('GPU')}")
logger.info('Experiment setup complete.')

# ==== Load and split metadata + features ====
train_data, nucleus_train, train_tag, val_data, nucleus_val, val_tag, test_data, nucleus_test, test_tag = data_setup.create_datasets(
    args.csv_file, args.qupath_feature_path, args.metadata_file, args.image_dir,
    args.nuc_feature, current_path, target=args.target, seed=args.seed,
    image_count=args.image_count, test_size=args.test_size
)

# ==== Logging basic shapes ====
logger.info("Shapes before patch processing:")
logger.info(f"Train: {nucleus_train.shape}, Val: {nucleus_val.shape}, Test: {nucleus_test.shape}")

# ==== Slide patch extraction ====
sum_x_train, sum_nucleus_train, sum_y_train, Image_ID_train, occup_train = processingslide.process_all_slides(train_data, nucleus_train, train_tag, True, args.image_dir)
sum_x_val, sum_nucleus_val, sum_y_val, Image_ID_val, occup_val = processingslide.process_all_slides(val_data, nucleus_val, val_tag, False, args.image_dir)
sum_x_test, sum_nucleus_test, sum_y_test, Image_ID_test, occup_test = processingslide.process_all_slides(test_data, nucleus_test, test_tag, False, args.image_dir)

# ==== Top patch selection per WSI ====
train_image, train_nucleus, train_labels, Image_ID_train_final, _ = processingslide.stack_data(sum_x_train, sum_nucleus_train, sum_y_train, Image_ID_train, occup_train)
val_image, val_nucleus, val_labels, Image_ID_val_final, _ = processingslide.stack_data(sum_x_val, sum_nucleus_val, sum_y_val, Image_ID_val, occup_val)
test_image, test_nucleus, test_labels, Image_ID_test_final, _ = processingslide.stack_data(sum_x_test, sum_nucleus_test, sum_y_test, Image_ID_test, occup_test)

# ==== Logging patch and sample counts ====
logger.info("Sample counts after stacking:")
logger.info(f"Train: {len(set(Image_ID_train_final))}, Val: {len(set(Image_ID_val_final))}, Test: {len(set(Image_ID_test_final))}")
logger.info("Image patch shape: %s", train_image.shape)

# ==== Split nucleus data into demographics + morphology ====
train_demographic = train_nucleus[:, :20]
val_demographic = val_nucleus[:, :20]
test_demographic = test_nucleus[:, :20]

train_nucleus_only = train_nucleus[:, 20:]
val_nucleus_only = val_nucleus[:, 20:]
test_nucleus_only = test_nucleus[:, 20:]

# ==== Build model ====
input_shape = train_image.shape[1:]
output_units = train_labels.shape[1]
feature_input_shape = train_nucleus.shape[1]

logger.info(f"Input image shape: {input_shape}")
logger.info(f"Output units: {output_units}")
logger.info(f"Nucleus feature dimensions: {feature_input_shape}")

model = model_builder.build_model(
    output_units,
    input_shape,
    demographic_feature_input_shape=20,
    nucleus_feature_input_shape=feature_input_shape - 20
)

logger.info("Model Summary:")
model.summary(print_fn=logger.info)

# ==== Optimizer + Callbacks ====
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=args.learning_rate,
    decay_steps=10000,
    decay_rate=0.95,
    staircase=True
)

callbacks = [
    ModelCheckpoint(filepath="noise_mnist.keras", save_best_only=True, monitor="val_loss"),
    EarlyStopping(monitor="val_loss", patience=30)
]

opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
model.compile(optimizer=opt, loss="mse")

# ==== Train model ====
history = model.fit(
    [train_image, train_demographic, train_nucleus_only],
    train_labels,
    epochs=args.epochs,
    batch_size=args.batch_size,
    validation_data=([val_image, val_demographic, val_nucleus_only], val_labels),
    callbacks=callbacks
)

# ==== Evaluate and predict ====
y_pred = model.predict([test_image, test_demographic, test_nucleus_only])
results = model.evaluate([test_image, test_demographic, test_nucleus_only], test_labels)

# ==== Save model and results ====
save_models.save_model(
    model=model,
    history=history,
    current_path=current_path,
    target_dir="models",
    model_name="basic.h5",
    y_pred=y_pred,
    test_label=test_labels
)

# ==== Generate evaluation plots ====
logger.info("Generating evaluation plots...")
plots.all_plots(history, y_pred, test_labels, Image_ID_test_final, current_path)
logger.info("Training complete.")
