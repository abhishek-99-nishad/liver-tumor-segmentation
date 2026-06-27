import tensorflow as tf
from tensorflow.keras import layers, models


def dense_block(x, filters, num_layers=2):
    """Lightweight Dense Block"""
    concat_features = [x]

    for _ in range(num_layers):
        y = layers.Conv2D(filters, 3, padding='same')(x)
        y = layers.BatchNormalization()(y)
        y = layers.Activation('relu')(y)

        concat_features.append(y)
        x = layers.Concatenate()(concat_features)

    return x


def transition_down(x, filters):
    """Downsampling"""
    x = layers.Conv2D(filters, 1, padding='same')(x)
    x = layers.MaxPooling2D()(x)
    return x


def transition_up(x, filters):
    """Upsampling"""
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(filters, 2, padding='same')(x)
    return x


def build_denseunet(input_shape=(256, 256, 1)):

    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = dense_block(inputs, 16)
    p1 = transition_down(c1, 16)

    c2 = dense_block(p1, 32)
    p2 = transition_down(c2, 32)

    c3 = dense_block(p2, 64)
    p3 = transition_down(c3, 64)

    c4 = dense_block(p3, 128)
    p4 = transition_down(c4, 128)

    # Bottleneck
    bn = dense_block(p4, 256)

    # Decoder
    u1 = transition_up(bn, 256)
    u1 = layers.Concatenate()([u1, c4])
    u1 = dense_block(u1, 256)

    u2 = transition_up(u1, 128)
    u2 = layers.Concatenate()([u2, c3])
    u2 = dense_block(u2, 128)

    u3 = transition_up(u2, 64)
    u3 = layers.Concatenate()([u3, c2])
    u3 = dense_block(u3, 64)

    u4 = transition_up(u3, 32)
    u4 = layers.Concatenate()([u4, c1])
    u4 = dense_block(u4, 32)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(u4)

    model = models.Model(inputs, outputs)

    return model