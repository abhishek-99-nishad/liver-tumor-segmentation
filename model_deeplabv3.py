from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2


def ASPP(x, filters):

    # 1x1 conv
    conv1 = layers.Conv2D(filters, 1, padding='same', activation='relu')(x)

    # Atrous conv blocks
    conv2 = layers.Conv2D(filters, 3, dilation_rate=6,
                          padding='same', activation='relu')(x)

    conv3 = layers.Conv2D(filters, 3, dilation_rate=12,
                          padding='same', activation='relu')(x)

    conv4 = layers.Conv2D(filters, 3, dilation_rate=18,
                          padding='same', activation='relu')(x)

    # Image pooling
    pool = layers.GlobalAveragePooling2D()(x)
    pool = layers.Reshape((1, 1, pool.shape[-1]))(pool)
    pool = layers.Conv2D(filters, 1, padding='same', activation='relu')(pool)
    pool = layers.UpSampling2D(
        size=(x.shape[1], x.shape[2]),
        interpolation="bilinear"
    )(pool)

    x = layers.Concatenate()([conv1, conv2, conv3, conv4, pool])
    x = layers.Conv2D(filters, 1, padding='same', activation='relu')(x)

    return x


def build_deeplabv3(input_shape=(256, 256, 1), trainable=False):

    inputs = layers.Input(shape=input_shape)

    # Convert 1-channel to 3-channel for MobileNet
    x = layers.Concatenate()([inputs, inputs, inputs])

    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_tensor=x
    )

    for layer in base_model.layers:
        layer.trainable = trainable

    # Use high-level feature map
    feature_map = base_model.get_layer("block_13_expand_relu").output

    # ASPP module
    x = ASPP(feature_map, 128)

    # Upsample to original size
    x = layers.UpSampling2D(size=(16, 16), interpolation="bilinear")(x)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)

    return models.Model(inputs, outputs)