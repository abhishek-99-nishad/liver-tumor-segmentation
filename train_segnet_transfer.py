import numpy as np
import tensorflow as tf
import wandb
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model_segnet import build_segnet  # Make sure this file exists

# Initialize wandb
wandb.init(project="LiverTumor", name="SegNet_TransferLearning")

# Load data
X_train = np.load("tumor_slices/training/ct_slices.npy")
y_train = np.load("tumor_slices/training/mask_slices.npy")
X_test = np.load("tumor_slices/testing/ct_slices.npy")
y_test = np.load("tumor_slices/testing/mask_slices.npy")

# Add channel dimension if needed
X_train = X_train[..., np.newaxis] if X_train.ndim == 3 else X_train
y_train = y_train[..., np.newaxis] if y_train.ndim == 3 else y_train
X_test = X_test[..., np.newaxis] if X_test.ndim == 3 else X_test
y_test = y_test[..., np.newaxis] if y_test.ndim == 3 else y_test

print("✅ X_train shape:", X_train.shape)
print("✅ y_train shape:", y_train.shape)

# Dice Loss function
def dice_loss(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1. - (2. * intersection + 1e-7) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1e-7)

# Build model (transfer learning = encoder frozen)
model = build_segnet(input_shape=(256, 256, 1), trainable=False)

# Compile model
model.compile(optimizer='adam',
              loss=lambda y_true, y_pred: tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred),
              metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("segnet_transfer_best_model.h5", monitor='val_loss', save_best_only=True)

# Train model (no WandbCallback used)
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=10,
                    batch_size=8,
                    callbacks=[early_stop, checkpoint])

# Manually log final metrics to wandb
wandb.log({
    "final_train_accuracy": history.history['accuracy'][-1],
    "final_val_accuracy": history.history['val_accuracy'][-1],
    "final_val_loss": history.history['val_loss'][-1]
})

# Save model
model.save("segnet_transfer_final_model.h5")
wandb.finish()
