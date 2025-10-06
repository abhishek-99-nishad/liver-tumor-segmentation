import numpy as np
import tensorflow as tf
import wandb
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model_segnet import build_segnet

# -----------------
# Initialize WandB
# -----------------

wandb.init(project="LiverTumor", name="SegNet_FineTuning")

# -----------------
# Load Data
# -----------------
X_train = np.load("tumor_slices/training/ct_slices.npy")
y_train = np.load("tumor_slices/training/mask_slices.npy")
X_val = np.load("tumor_slices/testing/ct_slices.npy")
y_val = np.load("tumor_slices/testing/mask_slices.npy")

# Expand dims if needed
if X_train.ndim == 3: X_train = X_train[..., np.newaxis]
if y_train.ndim == 3: y_train = y_train[..., np.newaxis]
if X_val.ndim == 3: X_val = X_val[..., np.newaxis]
if y_val.ndim == 3: y_val = y_val[..., np.newaxis]

print("✅ X_train shape:", X_train.shape)
print("✅ y_train shape:", y_train.shape)
print("✅ X_val shape:", X_val.shape)
print("✅ y_val shape:", y_val.shape)

# -----------------
# Dice Loss
# -----------------
def dice_loss(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1. - (2. * intersection + 1e-7) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1e-7
    )

# -----------------
# Build SegNet & Load Transfer Weights
# -----------------
model = build_segnet(trainable=True)  # fine-tuning → all layers trainable
model.load_weights("segnet_transfer_best_model.h5")

model.compile(
    optimizer='adam',
    loss=lambda y_true, y_pred:
        tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred),
    metrics=['accuracy']
)

# -----------------
# Callbacks
# -----------------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("segnet_finetune_best_model.h5", monitor='val_loss', save_best_only=True)

# -----------------
# Train
# -----------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=8,
    callbacks=[early_stop, checkpoint]
)

# -----------------
# Log Final Metrics
# -----------------
wandb.log({
    "final_train_accuracy": history.history['accuracy'][-1],
    "final_val_accuracy": history.history['val_accuracy'][-1],
    "final_val_loss": history.history['val_loss'][-1]
})

# -----------------
# Save Model
# -----------------
model.save("segnet_finetune_final_model.h5")
wandb.finish()
