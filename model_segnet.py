import tensorflow as tf

def build_segnet(input_shape=(256, 256, 1), trainable=True):
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', trainable=trainable)(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu', trainable=trainable)(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu', trainable=trainable)(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu', trainable=trainable)(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    # Decoder
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu', trainable=trainable)(x)

    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu', trainable=trainable)(x)

    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu', trainable=trainable)(x)

    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', trainable=trainable)(x)

    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(x)

    return tf.keras.Model(inputs, outputs)
