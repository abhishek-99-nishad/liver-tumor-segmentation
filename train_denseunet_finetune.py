import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import wandb
from wandb.integration.keras import WandbCallback

from model_denseunet import build_denseunet


# Initialize wandb
wandb.init(project="LiverTumor", name="DenseUNet_Finetune")


# Load data
X_train = np.load("tumor_slices/training/ct_slices.npy")
y_train = np.load("tumor_slices/training/mask_slices.npy")

X_val = np.load("tumor_slices/testing/ct_slices.npy")
y_val = np.load("tumor_slices/testing/mask_slices.npy")


# Dice Loss
def dice_loss(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


# Combined Loss
def combined_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


# Build model (now fully trainable — same as transfer, but we increase learning capacity)
model = build_denseunet(input_shape=(256, 256, 1))


# Compile (slightly lower LR for finetuning stability)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=combined_loss,
    metrics=["accuracy"]
)


# Callbacks
checkpoint = ModelCheckpoint(
    "denseunet_finetune_best_model.h5",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

earlystop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)


# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=4,
    callbacks=[checkpoint, earlystop, WandbCallback()],
    verbose=1
)


# Save final model
model.save("denseunet_finetune_final_model.h5")