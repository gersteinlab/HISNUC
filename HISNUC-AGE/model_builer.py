import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import random

# Ensure reproducibility
keras.utils.set_random_seed(20)
tf.config.experimental.enable_op_determinism()

def build_selfnorm_encoder(input_dim=20, hidden_dim=12, output_dim=6, dropout_rate=0.1):
    """
    Builds a self-normalizing encoder using SELU and AlphaDropout layers.
    Used for encoding either demographic or nucleus features.
    """
    return models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(hidden_dim, activation='selu', kernel_initializer='lecun_normal'),
        layers.AlphaDropout(rate=dropout_rate),
        layers.Dense(output_dim, activation='selu', kernel_initializer='lecun_normal'),
        layers.AlphaDropout(rate=dropout_rate)
    ])

def build_model(output_units: int, input_shape: tuple, demographic_feature_input_shape: int, nucleus_feature_input_shape: int):
    """
    Builds the full model with:
    - CNN backbone for image patch encoding
    - Self-normalizing encoders for demographic and nucleus features
    - Outer product fusion of modalities
    - Fully connected regression head for output

    Parameters:
    - output_units: number of genes/targets to predict
    - input_shape: shape of each image patch (H, W, C)
    - demographic_feature_input_shape: number of demographic input features
    - nucleus_feature_input_shape: number of nucleus input features

    Returns:
    - A compiled tf.keras.Model
    """
    random.seed(10)

    # --- Image branch (CNN) ---
    image_input = layers.Input(shape=input_shape, name='image_input')
    x = layers.Conv2D(64, (3, 3), activation='relu')(image_input)
    x = layers.AveragePooling2D()(x)
    x = layers.Dropout(0.2, seed=20)(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.AveragePooling2D()(x)
    x = layers.Dropout(0.2, seed=20)(x)
    x = layers.Conv2D(256, (3, 3), activation='relu')(x)
    x = layers.AveragePooling2D()(x)
    x = layers.Dropout(0.2, seed=20)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)

    # --- Metadata encoders ---
    demg_input = layers.Input(shape=(demographic_feature_input_shape,), name='demographic_input')
    nuc_input = layers.Input(shape=(nucleus_feature_input_shape,), name='nucleus_input')

    encoded_demg = build_selfnorm_encoder(demographic_feature_input_shape)(demg_input)
    encoded_nuc = build_selfnorm_encoder(nucleus_feature_input_shape)(nuc_input)

    # --- Outer product fusion with broadcasting ---
    x_with_bias = tf.concat([x, tf.ones_like(x[:, :1])], axis=-1)
    d_with_bias = tf.concat([encoded_demg, tf.ones_like(encoded_demg[:, :1])], axis=-1)
    n_with_bias = tf.concat([encoded_nuc, tf.ones_like(encoded_nuc[:, :1])], axis=-1)

    x_exp = tf.expand_dims(tf.expand_dims(x_with_bias, 2), 3)  # (batch, F, 1, 1)
    d_exp = tf.expand_dims(tf.expand_dims(d_with_bias, 1), 3)  # (batch, 1, D, 1)
    n_exp = tf.expand_dims(tf.expand_dims(n_with_bias, 1), 2)  # (batch, 1, 1, N)

    fusion_tensor = x_exp * d_exp * n_exp
    fusion_flat = tf.reshape(fusion_tensor, shape=(-1, 65 * 7 * 7))  # Flatten

    # --- Fully connected prediction head ---
    z = layers.Dense(256, activation='relu')(fusion_flat)
    z = layers.Dense(16, activation='relu')(z)
    output = layers.Dense(output_units, activation='linear')(z)

    model = keras.Model(inputs=[image_input, demg_input, nuc_input], outputs=output)
    return model
