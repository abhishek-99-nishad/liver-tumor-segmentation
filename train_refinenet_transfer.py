import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import wandb
from model_refinenet import build_refinenet

# Init wandb
wandb.init(project="LiverTumor", name="RefineNet_TransferLearning")

# Load data
X_train = np.load("tumor_slices/training/ct_slices.npy")[..., np.newaxis]
y_train = np.load("tumor_slices/training/mask_slices.npy")[..., np.newaxis]
X_val = np.load("tumor_slices/testing/ct_slices.npy")[..., np.newaxis]
y_val = np.load("tumor_slices/testing/mask_slices.npy")[..., np.newaxis]

# Dice Loss
def dice_loss(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1. - (2. * intersection + 1e-7) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1e-7)

# Build & compile (encoder frozen)
model = build_refinenet(trainable=False)
model.compile(optimizer="adam",
              loss=lambda y_true, y_pred: tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred),
              metrics=["accuracy"])

# Callbacks
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("refinenet_transfer_best_model.h5", monitor="val_loss", save_best_only=True)

# Train
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=10,
                    batch_size=8,
                    callbacks=[early_stop, checkpoint])

# Log final metrics
wandb.log({
    "final_train_accuracy": history.history["accuracy"][-1],
    "final_val_accuracy": history.history["val_accuracy"][-1],
    "final_val_loss": history.history["val_loss"][-1]
})

model.save("refinenet_transfer_final_model.h5")
wandb.finish()
