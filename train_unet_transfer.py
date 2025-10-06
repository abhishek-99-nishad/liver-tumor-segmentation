import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import wandb

# Initialize wandb
wandb.init(project="LiverTumor", name="U-Net_TransferLearning")

# Load data
X_train = np.load("tumor_slices/training/ct_slices.npy")
y_train = np.load("tumor_slices/training/mask_slices.npy")
X_test = np.load("tumor_slices/testing/ct_slices.npy")
y_test = np.load("tumor_slices/testing/mask_slices.npy")

print("✅ X shape:", X_train.shape)
print("✅ y shape:", y_train.shape)

# Dice Loss
def dice_loss(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1. - (2. * intersection + 1e-7) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1e-7)

# U-Net Model
def build_unet(input_shape=(256, 256, 1)):
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    c1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D()(c1)

    c2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D()(c2)

    c3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D()(c3)

    # Bottleneck
    b = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(p3)
    b = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(b)

    # Decoder
    u3 = tf.keras.layers.UpSampling2D()(b)
    u3 = tf.keras.layers.concatenate([u3, c3])
    c6 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(u3)
    c6 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(c6)

    u2 = tf.keras.layers.UpSampling2D()(c6)
    u2 = tf.keras.layers.concatenate([u2, c2])
    c7 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(u2)
    c7 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(c7)

    u1 = tf.keras.layers.UpSampling2D()(c7)
    u1 = tf.keras.layers.concatenate([u1, c1])
    c8 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(u1)
    c8 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(c8)

    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(c8)

    model = tf.keras.Model(inputs, outputs)
    return model

# Compile Model
model = build_unet()
model.compile(optimizer='adam',
              loss=lambda y_true, y_pred: tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred),
              metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("unet_transfer_best_model.h5", monitor='val_loss', save_best_only=True)

# Train (no WandbCallback here!)
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=10,
                    batch_size=8,
                    callbacks=[early_stop, checkpoint])

# Manually log final accuracy and loss to wandb
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
final_val_loss = history.history['val_loss'][-1]

wandb.log({
    "final_train_accuracy": final_train_acc,
    "final_val_accuracy": final_val_acc,
    "final_val_loss": final_val_loss
})

# Save final model (optional)
model.save("unet_transfer_final_model.h5")
wandb.finish()
