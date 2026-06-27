from tensorflow.keras import layers, models
import tensorflow as tf


def attention_gate(skip, gating, inter_channels):
    """
    Lightweight attention gate compatible with your shallow U-Net
    """
    theta_x = layers.Conv2D(inter_channels, 1, padding='same')(skip)
    phi_g = layers.Conv2D(inter_channels, 1, padding='same')(gating)

    add = layers.Add()([theta_x, phi_g])
    relu = layers.Activation('relu')(add)

    psi = layers.Conv2D(1, 1, padding='same')(relu)
    sigmoid = layers.Activation('sigmoid')(psi)

    out = layers.Multiply()([skip, sigmoid])
    return out


def build_attention_unet(input_shape=(256, 256, 1), pretrained_encoder=None, trainable=False):
    inputs = layers.Input(shape=input_shape)

    # ---------------- Encoder (optionally pretrained) ----------------
    if pretrained_encoder:
        base_model = pretrained_encoder(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs
        )
        for layer in base_model.layers:
            layer.trainable = trainable

        x = base_model.output
        skip_connection = None  # pretrained path has no simple skip here

    else:
        # Simple encoder (same as your U-Net)
        c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
        c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
        p1 = layers.MaxPooling2D()(c1)

        c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
        c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
        p2 = layers.MaxPooling2D()(c2)

        x = p2
        skip_connection = c1  # attention will act here
 
    # ---------------- Decoder ----------------
    u1 = layers.UpSampling2D()(x)
    u1 = layers.Conv2D(64, 3, activation='relu', padding='same')(u1)

    # -------- Attention on skip connection (ONLY difference) --------
    if skip_connection is not None:
        attn = attention_gate(skip_connection, u1, inter_channels=32)
        u1 = layers.Concatenate()([u1, attn])

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(u1)

    return models.Model(inputs, outputs)
