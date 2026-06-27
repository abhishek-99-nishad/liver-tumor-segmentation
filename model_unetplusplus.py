import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D,
    Concatenate, BatchNormalization, Activation
)
from tensorflow.keras.models import Model


def conv_block(x, filters):
    x = Conv2D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def build_unetplusplus(input_shape=(256, 256, 1)):
    inputs = Input(input_shape)

    # Encoder
    x00 = conv_block(inputs, 16)
    p0 = MaxPooling2D((2, 2))(x00)

    x10 = conv_block(p0, 32)
    p1 = MaxPooling2D((2, 2))(x10)

    x20 = conv_block(p1, 64)
    p2 = MaxPooling2D((2, 2))(x20)

    x30 = conv_block(p2, 128)
    p3 = MaxPooling2D((2, 2))(x30)

    x40 = conv_block(p3, 256)

    # Decoder (nested skip connections)
    x01 = conv_block(Concatenate()([x00, UpSampling2D((2, 2))(x10)]), 16)

    x11 = conv_block(Concatenate()([x10, UpSampling2D((2, 2))(x20)]), 32)

    x21 = conv_block(Concatenate()([x20, UpSampling2D((2, 2))(x30)]), 64)

    x31 = conv_block(Concatenate()([x30, UpSampling2D((2, 2))(x40)]), 128)

    x02 = conv_block(
        Concatenate()([x00, x01, UpSampling2D((2, 2))(x11)]), 16
    )

    x12 = conv_block(
        Concatenate()([x10, x11, UpSampling2D((2, 2))(x21)]), 32
    )

    x22 = conv_block(
        Concatenate()([x20, x21, UpSampling2D((2, 2))(x31)]), 64
    )

    x03 = conv_block(
        Concatenate()([x00, x01, x02, UpSampling2D((2, 2))(x12)]), 16
    )

    x13 = conv_block(
        Concatenate()([x10, x11, x12, UpSampling2D((2, 2))(x22)]), 32
    )

    x04 = conv_block(
        Concatenate()([x00, x01, x02, x03, UpSampling2D((2, 2))(x13)]), 16
    )

    outputs = Conv2D(1, 1, activation="sigmoid")(x04)

    model = Model(inputs, outputs, name="UNetPlusPlus")

    return model