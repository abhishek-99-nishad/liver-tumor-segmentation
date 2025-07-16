import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import Precision, Recall

#  Dice Loss
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

#  Combined BCE + Dice Loss
def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

#  Load preprocessed data
DATA_DIR = "preprocessed"
all_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith("_ct.npy")])
X, Y = [], []

for ct_file in all_files:
    mask_file = ct_file.replace("_ct.npy", "_mask.npy")
    ct = np.load(os.path.join(DATA_DIR, ct_file))
    mask = np.load(os.path.join(DATA_DIR, mask_file))
    X.extend(ct)
    Y.extend(mask)

X = np.expand_dims(np.array(X), axis=-1)
Y = np.expand_dims(np.array(Y), axis=-1)

print(f"✅ Loaded {X.shape[0]} slices.")

#  Train/Val Split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

#  SegNet Model Definition
def build_segnet(input_shape=(256, 256, 1)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    # Decoder
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)

    return models.Model(inputs, outputs)

#  Compile SegNet
model = build_segnet()
model.compile(
    optimizer='adam',
    loss=bce_dice_loss,
    metrics=['accuracy', Precision(), Recall()]
)

model.summary()

#  Callbacks
checkpoint = ModelCheckpoint("segnet_model_best.h5", monitor='val_loss', save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

#  Train
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=10,
    batch_size=8,
    callbacks=[checkpoint, early_stop]
)

#  Final F1 Score
precision = history.history['precision'][-1]
recall = history.history['recall'][-1]
f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
print(f"🔍 Final F1 Score: {f1:.4f}")

#  Save final model
model.save("segnet_model.h5")
print("✅ Final model saved as segnet_model.h5")

#  Plot Loss Curve
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.title("SegNet Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("segnet_loss.png")
plt.show()

#  Plot Training Metrics
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Val Accuracy")
plt.plot(history.history['precision'], label="Precision")
plt.plot(history.history['recall'], label="Recall")
plt.title("SegNet Training Metrics")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.savefig("segnet_metrics.png")
plt.show()
