from tensorflow.keras import layers, models

def build_unet(input_shape=(256, 256, 1), pretrained_encoder=None, trainable=False):
    inputs = layers.Input(shape=input_shape)

    # Encoder (optionally pretrained)
    if pretrained_encoder:
        base_model = pretrained_encoder(include_top=False, weights='imagenet', input_tensor=inputs)
        for layer in base_model.layers:
            layer.trainable = trainable
        x = base_model.output
    else:
        # Simple encoder
        c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
        c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
        p1 = layers.MaxPooling2D()(c1)

        c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
        c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
        p2 = layers.MaxPooling2D()(c2)

        x = p2  # continue building decoder from here

    # Decoder (example block)
    u1 = layers.UpSampling2D()(x)
    u1 = layers.Conv2D(64, 3, activation='relu', padding='same')(u1)
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(u1)

    return models.Model(inputs, outputs)
