from tensorflow.keras import layers, models


def residual_block(x, filters):
    shortcut = x

    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding='same')(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)

    return x


def build_resunet(input_shape=(256, 256, 1), trainable=False):

    inputs = layers.Input(shape=input_shape)

    # ↓↓↓ Reduced filters by half
    c1 = residual_block(inputs, 32)
    p1 = layers.MaxPooling2D()(c1)

    c2 = residual_block(p1, 64)
    p2 = layers.MaxPooling2D()(c2)

    c3 = residual_block(p2, 128)
    p3 = layers.MaxPooling2D()(c3)

    c4 = residual_block(p3, 256)
    p4 = layers.MaxPooling2D()(c4)

    # ↓↓↓ Bottleneck reduced
    x = residual_block(p4, 512)

    # Decoder
    u1 = layers.UpSampling2D()(x)
    u1 = layers.Concatenate()([u1, c4])
    u1 = residual_block(u1, 256)

    u2 = layers.UpSampling2D()(u1)
    u2 = layers.Concatenate()([u2, c3])
    u2 = residual_block(u2, 128)

    u3 = layers.UpSampling2D()(u2)
    u3 = layers.Concatenate()([u3, c2])
    u3 = residual_block(u3, 64)

    u4 = layers.UpSampling2D()(u3)
    u4 = layers.Concatenate()([u4, c1])
    u4 = residual_block(u4, 32)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(u4)

    return models.Model(inputs, outputs)