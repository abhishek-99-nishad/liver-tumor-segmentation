import numpy as np
import tensorflow as tf
import wandb
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Initialize wandb
wandb.init(project="LiverTumor", name="AttentionUNet_FineTuning")

# Load data
X_train = np.load("tumor_slices/training/ct_slices.npy")
y_train = np.load("tumor_slices/training/mask_slices.npy")
X_val = np.load("tumor_slices/testing/ct_slices.npy")
y_val = np.load("tumor_slices/testing/mask_slices.npy")

# Expand dims if needed
X_train = X_train[..., np.newaxis] if X_train.ndim == 3 else X_train
y_train = y_train[..., np.newaxis] if y_train.ndim == 3 else y_train
X_val = X_val[..., np.newaxis] if X_val.ndim == 3 else X_val
y_val = y_val[..., np.newaxis] if y_val.ndim == 3 else y_val

print("✅ X_train:", X_train.shape)
print("✅ y_train:", y_train.shape)

# -----------------------------
# Dice Loss
# -----------------------------
def dice_loss(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1. - (2. * intersection + 1e-7) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1e-7
    )

# -----------------------------
# Attention Gate
# -----------------------------
def attention_gate(x, g, inter_channels):
    theta_x = tf.keras.layers.Conv2D(inter_channels, 1, padding='same')(x)
    phi_g = tf.keras.layers.Conv2D(inter_channels, 1, padding='same')(g)

    add = tf.keras.layers.Add()([theta_x, phi_g])
    act = tf.keras.layers.Activation('relu')(add)

    psi = tf.keras.layers.Conv2D(1, 1, padding='same')(act)
    psi = tf.keras.layers.Activation('sigmoid')(psi)

    return tf.keras.layers.Multiply()([x, psi])

# -----------------------------
# Build Attention U-Net (Fine-Tune)
# -----------------------------
def build_attention_unet(input_shape=(256, 256, 1), trainable=True):

    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    c1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', trainable=trainable)(inputs)
    c1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', trainable=trainable)(c1)
    p1 = tf.keras.layers.MaxPooling2D()(c1)

    c2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', trainable=trainable)(p1)
    c2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', trainable=trainable)(c2)
    p2 = tf.keras.layers.MaxPooling2D()(c2)

    c3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', trainable=trainable)(p2)
    c3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', trainable=trainable)(c3)
    p3 = tf.keras.layers.MaxPooling2D()(c3)

    # Bottleneck
    b = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', trainable=trainable)(p3)
    b = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', trainable=trainable)(b)

    # Decoder with Attention
    u3 = tf.keras.layers.UpSampling2D()(b)
    a3 = attention_gate(c3, u3, inter_channels=128)
    u3 = tf.keras.layers.concatenate([u3, a3])
    c6 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', trainable=trainable)(u3)
    c6 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', trainable=trainable)(c6)

    u2 = tf.keras.layers.UpSampling2D()(c6)
    a2 = attention_gate(c2, u2, inter_channels=64)
    u2 = tf.keras.layers.concatenate([u2, a2])
    c7 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', trainable=trainable)(u2)
    c7 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', trainable=trainable)(c7)

    u1 = tf.keras.layers.UpSampling2D()(c7)
    a1 = attention_gate(c1, u1, inter_channels=32)
    u1 = tf.keras.layers.concatenate([u1, a1])
    c8 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', trainable=trainable)(u1)
    c8 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', trainable=trainable)(c8)

    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(c8)

    return tf.keras.Model(inputs, outputs)

# -----------------------------
# Compile model
# -----------------------------
model = build_attention_unet(trainable=True)

# IMPORTANT:
# Fine-tuning assumes you already trained Attention U-Net with transfer learning
model.load_weights("attention_unet_transfer_best_model.h5")

model.compile(
    optimizer='adam',
    loss=lambda y_true, y_pred: tf.keras.losses.binary_crossentropy(y_true, y_pred)
                                + dice_loss(y_true, y_pred),
    metrics=['accuracy']
)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(
    "attention_unet_finetune_best_model.h5",
    monitor='val_loss',
    save_best_only=True
)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=8,
    callbacks=[early_stop, checkpoint]
)

# Log to wandb
wandb.log({
    "final_train_accuracy": history.history['accuracy'][-1],
    "final_val_accuracy": history.history['val_accuracy'][-1],
    "final_val_loss": history.history['val_loss'][-1]
})

# Save final model
model.save("attention_unet_finetune_final_model.h5")
wandb.finish()
