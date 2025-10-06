import tensorflow as tf

def conv_block(x, filters, trainable=True):
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False, trainable=trainable)(x)
    x = tf.keras.layers.BatchNormalization(trainable=trainable)(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x

def refine_block(high_res, low_res, filters, trainable=True):
    """Refinement block: fuse low + high resolution features"""
    low_res = tf.keras.layers.UpSampling2D(size=(2, 2))(low_res)
    low_res = conv_block(low_res, filters, trainable=trainable)
    high_res = conv_block(high_res, filters, trainable=trainable)
    return tf.keras.layers.Add()([high_res, low_res])

def build_refinenet(input_shape=(256, 256, 1), trainable=True):
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder (ResNet-like but lighter)
    c1 = conv_block(inputs, 64, trainable)
    p1 = tf.keras.layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 128, trainable)
    p2 = tf.keras.layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 256, trainable)
    p3 = tf.keras.layers.MaxPooling2D()(c3)

    c4 = conv_block(p3, 512, trainable)

    # RefineNet Decoder
    r3 = refine_block(c3, c4, 256, trainable)
    r2 = refine_block(c2, r3, 128, trainable)
    r1 = refine_block(c1, r2, 64, trainable)

    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(r1)

    return tf.keras.Model(inputs, outputs)
