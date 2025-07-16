import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# --- Dice + BCE loss ---
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

# --- Load Data ---
DATA_DIR = "preprocessed"
ct_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith("_ct.npy")])
X, Y = [], []

for ct_file in ct_files:
    ct = np.load(os.path.join(DATA_DIR, ct_file))
    mask_file = ct_file.replace("_ct.npy", "_mask.npy")
    mask = np.load(os.path.join(DATA_DIR, mask_file))
    X.extend(ct)
    Y.extend(mask)

X = np.expand_dims(np.array(X), axis=-1)
Y = np.expand_dims(np.array(Y), axis=-1)
print(f"✅ Loaded {X.shape[0]} slices")

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# --- RefineNet-like Architecture (simplified) ---
def build_refinenet(input_shape=(256, 256, 1)):
    inputs = tf.keras.Input(shape=input_shape)

    def residual_block(x, filters):
        shortcut = layers.Conv2D(filters, 1, padding='same')(x)
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        return x

    def fuse_block(x1, x2, filters):
        x1 = layers.UpSampling2D()(x1)
        x = layers.Concatenate()([x1, x2])
        x = residual_block(x, filters)
        return x

    # Encoder
    x1 = residual_block(inputs, 32)
    p1 = layers.MaxPooling2D()(x1)

    x2 = residual_block(p1, 64)
    p2 = layers.MaxPooling2D()(x2)

    x3 = residual_block(p2, 128)
    p3 = layers.MaxPooling2D()(x3)

    # Bottleneck
    bn = residual_block(p3, 256)

    # Decoder (Refine-style fusion)
    d3 = fuse_block(bn, x3, 128)
    d2 = fuse_block(d3, x2, 64)
    d1 = fuse_block(d2, x1, 32)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(d1)
    return models.Model(inputs, outputs)

# --- Compile & Train ---
model = build_refinenet()
model.compile(optimizer='adam', loss=bce_dice_loss, metrics=['accuracy', Precision(), Recall()])
model.summary()

checkpoint = ModelCheckpoint("refinenet_model.h5", monitor='val_loss', save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=10,
    batch_size=8,
    callbacks=[checkpoint, early_stop]
)

# --- Final F1 Score ---
precision = history.history['precision'][-1]
recall = history.history['recall'][-1]
f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
print(f"🔍 Final F1 Score: {f1:.4f}")

# --- Save final model ---
model.save("refinenet_model_final.h5")
print("✅ Saved final RefineNet model")

# --- Plot Loss Curve ---
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.title("RefineNet Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("refinenet_loss.png")
plt.show()
