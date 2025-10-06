import numpy as np
import tensorflow as tf
import wandb
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ---------- Initialize W&B ----------
wandb.init(project="LiverTumor", name="MA-Net_FineTuning")

# ---------- Load data ----------
X_train = np.load("tumor_slices/training/ct_slices.npy")
y_train = np.load("tumor_slices/training/mask_slices.npy")
X_val   = np.load("tumor_slices/testing/ct_slices.npy")
y_val   = np.load("tumor_slices/testing/mask_slices.npy")

# ---------- Expand channel dim if needed ----------
X_train = X_train[..., np.newaxis] if X_train.ndim == 3 else X_train
y_train = y_train[..., np.newaxis] if y_train.ndim == 3 else y_train
X_val   = X_val[..., np.newaxis]   if X_val.ndim   == 3 else X_val
y_val   = y_val[..., np.newaxis]   if y_val.ndim   == 3 else y_val

# ---------- Dice loss ----------
def dice_loss(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1. - (2. * intersection + 1e-7) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1e-7)

# ---------- Build MANet and load transfer weights ----------
from manet_builder import build_manet  # or place the builder above and import locally

model = build_manet(input_shape=X_train.shape[1:], trainable=True, num_classes=1)
model.load_weights("manet_transfer_best_model.h5")  # swap to your transfer checkpoint

# ---------- Compile ----------
model.compile(
    optimizer='adam',
    loss=lambda yt, yp: tf.keras.losses.binary_crossentropy(yt, yp) + dice_loss(yt, yp),
    metrics=['accuracy']
)

# ---------- Callbacks ----------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("manet_finetune_best_model.h5", monitor='val_loss', save_best_only=True)

# ---------- Train ----------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=16,
    callbacks=[early_stop, checkpoint]
)

# ---------- Log to W&B ----------
wandb.log({
    "final_train_accuracy": history.history['accuracy'][-1],
    "final_val_accuracy": history.history['val_accuracy'][-1],
    "final_val_loss": history.history['val_loss'][-1]
})

# ---------- Save final ----------
model.save("manet_finetune_final_model.h5")
wandb.finish()
