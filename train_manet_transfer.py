import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# No W&B callback here (kept identical to your UNet transfer setup)

# ---------- Load data ----------
X_train = np.load("tumor_slices/training/ct_slices.npy")
y_train = np.load("tumor_slices/training/mask_slices.npy")
X_test  = np.load("tumor_slices/testing/ct_slices.npy")
y_test  = np.load("tumor_slices/testing/mask_slices.npy")

print("✅ X shape:", X_train.shape)
print("✅ y shape:", y_train.shape)

# Ensure channel dimension exists (H, W, 1)
X_train = X_train[..., np.newaxis] if X_train.ndim == 3 else X_train
y_train = y_train[..., np.newaxis] if y_train.ndim == 3 else y_train
X_test  = X_test[...,  np.newaxis] if X_test.ndim  == 3 else X_test
y_test  = y_test[...,  np.newaxis] if y_test.ndim  == 3 else y_test

# ---------- Dice loss (same as UNet transfer) ----------
def dice_loss(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1. - (2. * intersection + 1e-7) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1e-7)

# ---------- MANet builder (inline for single-file usage) ----------
from tensorflow.keras import layers, models

def se_block(x, reduction=16):
    ch = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(max(ch // reduction, 1), activation='relu')(se)
    se = layers.Dense(ch, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, ch))(se)
    return layers.Multiply()([x, se])

def spatial_attention(x, kernel_size=7):
    avg_pool = layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True))(x)
    max_pool = layers.Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True))(x)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    attn = layers.Conv2D(1, kernel_size, padding='same', activation='sigmoid')(concat)
    return layers.Multiply()([x, attn])

def residual_block(x, filters, stride=1):
    shortcut = x
    y = layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False)(x)
    y = layers.BatchNormalization()(y); y = layers.ReLU()(y)
    y = layers.Conv2D(filters, 3, padding='same', use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    y = layers.Add()([shortcut, y]); y = layers.ReLU()(y)
    return y

def mfab(x, filters):
    b1 = layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
    b1 = layers.BatchNormalization()(b1); b1 = layers.ReLU()(b1)
    b2 = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    b2 = layers.BatchNormalization()(b2); b2 = layers.ReLU()(b2)
    b3 = layers.Conv2D(filters, 5, padding='same', use_bias=False)(x)
    b3 = layers.BatchNormalization()(b3); b3 = layers.ReLU()(b3)
    m = layers.Add()([b1, b2, b3])
    m = se_block(m)
    return m

def pab(x, filters):
    m = mfab(x, filters)
    m = spatial_attention(m)
    return m

def decoder_block(x, skip, filters):
    x = layers.UpSampling2D()(x)
    if skip is not None:
        s = spatial_attention(skip)
        x = layers.Concatenate()([x, s])
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = se_block(x)
    return x

def build_manet(input_shape=(256, 256, 1), num_classes=1):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    e1 = residual_block(inputs, 64, stride=1)
    e1 = pab(e1, 64); p1 = layers.MaxPooling2D()(e1)

    e2 = residual_block(p1, 128, stride=1)
    e2 = pab(e2, 128); p2 = layers.MaxPooling2D()(e2)

    e3 = residual_block(p2, 256, stride=1)
    e3 = pab(e3, 256); p3 = layers.MaxPooling2D()(e3)

    # Bottleneck
    b = residual_block(p3, 512, stride=1)
    b = pab(b, 512)

    # Decoder
    d3 = decoder_block(b, e3, 256)
    d2 = decoder_block(d3, e2, 128)
    d1 = decoder_block(d2, e1, 64)

    activation = 'sigmoid' if num_classes == 1 else 'softmax'
    outputs = layers.Conv2D(num_classes, 1, activation=activation)(d1)
    return models.Model(inputs, outputs)

# ---------- Build, compile, callbacks, train ----------
model = build_manet(input_shape=X_train.shape[1:], num_classes=1)

model.compile(
    optimizer='adam',
    loss=lambda yt, yp: tf.keras.losses.binary_crossentropy(yt, yp) + dice_loss(yt, yp),
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("manet_transfer_best_model.h5", monitor='val_loss', save_best_only=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=5,
    batch_size=16,
    callbacks=[early_stop, checkpoint]
)

# ---------- Save final model ----------
model.save("manet_transfer_final_model.h5")
