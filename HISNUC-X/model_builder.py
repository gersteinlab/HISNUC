
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from keras.layers import Dropout, Concatenate
from keras.regularizers import l2

# Set deterministic behavior and seed
keras.utils.set_random_seed(20)
tf.config.experimental.enable_op_determinism()


def build_selfnorm_encoder(input_dim=25, hidden_dim=22, output_dim=20, dropout_rate=0.05):
    """
    Build a self-normalizing encoder using SELU and AlphaDropout.
    """
    return models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(hidden_dim, activation='selu', kernel_initializer='lecun_normal'),
        layers.AlphaDropout(rate=dropout_rate),
        layers.Dense(output_dim, activation='selu', kernel_initializer='lecun_normal'),
        layers.AlphaDropout(rate=dropout_rate)
    ])


def build_selfnorm_encoder2(input_dim=20, hidden_dim=16, output_dim=12, dropout_rate=0.05):
    """
    A second variant of the self-normalizing encoder with different dimensions.
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
    Constructs the full model including:
    - CNN for image features
    - Self-normalizing encoders for metadata
    - Outer product fusion of image, demographic, and nucleus features
    - Dense layers for final prediction

    Parameters:
    - output_units: Number of output units (e.g., genes to predict)
    - input_shape: Shape of the image input (e.g., (32, 32, 128))
    - demographic_feature_input_shape: Number of demographic input features
    - nucleus_feature_input_shape: Number of nucleus input features

    Returns:
    - tf.keras.Model: The constructed model
    """
    random.seed(10)

    # Image CNN subnetwork
    image_input = layers.Input(shape=input_shape, name='image_input')
    x = layers.Conv2D(64, kernel_size=3, activation='relu')(image_input)
    x = layers.AveragePooling2D()(x)
    x = layers.Dropout(0.2, seed=20)(x)
    x = layers.Conv2D(128, kernel_size=3, activation='relu')(x)
    x = layers.AveragePooling2D()(x)
    x = layers.Dropout(0.2, seed=20)(x)
    x = layers.Conv2D(256, kernel_size=3, activation='relu')(x)
    x = layers.AveragePooling2D()(x)
    x = layers.Dropout(0.2, seed=20)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)

    # Metadata encoders
    demg_input = layers.Input(shape=(demographic_feature_input_shape,), name='demographic_input')
    nuc_input = layers.Input(shape=(nucleus_feature_input_shape,), name='nucleus_input')

    encoded_demg = build_selfnorm_encoder(demographic_feature_input_shape)(demg_input)
    encoded_nuc = build_selfnorm_encoder2(nucleus_feature_input_shape)(nuc_input)

    # Outer product fusion (using broadcasting)
    pf_with_bias = tf.concat([x, tf.ones_like(x[:, :1])], axis=-1)  # shape: (batch, 65)
    ed_with_bias = tf.concat([encoded_demg, tf.ones_like(encoded_demg[:, :1])], axis=-1)  # shape: (batch, 21)
    en_with_bias = tf.concat([encoded_nuc, tf.ones_like(encoded_nuc[:, :1])], axis=-1)  # shape: (batch, 13)

    pf_exp = tf.expand_dims(tf.expand_dims(pf_with_bias, axis=2), axis=3)
    ed_exp = tf.expand_dims(tf.expand_dims(ed_with_bias, axis=1), axis=3)
    en_exp = tf.expand_dims(tf.expand_dims(en_with_bias, axis=1), axis=2)

    fusion = pf_exp * ed_exp * en_exp  # outer product
    fusion_flat = tf.reshape(fusion, shape=(-1, 65 * 21 * 13))

    # Post-fusion dense layers
    z = layers.Dense(512, activation='relu')(fusion_flat)
    output = layers.Dense(output_units, activation='linear')(z)

    return keras.Model(inputs=[image_input, demg_input, nuc_input], outputs=output)
