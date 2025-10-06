import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ✅ Custom Dice Loss Function
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

# ✅ Combined BCE + Dice Loss
def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

# ✅ Load preprocessed data
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

# ✅ Split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# ✅ Build U-Net
def build_unet(input_shape=(256, 256, 1)):
    inputs = tf.keras.Input(input_shape)

    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
        x = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
        return x

    def encoder_block(x, filters):
        c = conv_block(x, filters)
        p = layers.MaxPooling2D()(c)
        return c, p

    def decoder_block(x, skip, filters):
        x = layers.Conv2DTranspose(filters, 2, strides=2, padding='same')(x)
        x = layers.Concatenate()([x, skip])
        return conv_block(x, filters)

    # Encoder
    c1, p1 = encoder_block(inputs, 32)
    c2, p2 = encoder_block(p1, 64)
    c3, p3 = encoder_block(p2, 128)

    # Bottleneck
    bn = conv_block(p3, 256)

    # Decoder
    d3 = decoder_block(bn, c3, 128)
    d2 = decoder_block(d3, c2, 64)
    d1 = decoder_block(d2, c1, 32)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(d1)

    return models.Model(inputs, outputs)

# ✅ Compile Model
model = build_unet()
model.compile(
    optimizer='adam',
    loss=bce_dice_loss,
    metrics=['accuracy', Precision(), Recall()]
)
model.summary()

# ✅ Callbacks
checkpoint = ModelCheckpoint("unet_model_best.h5", monitor='val_loss', save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# ✅ Train Model
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=10,
    batch_size=8,
    callbacks=[checkpoint, early_stop]
)

# ✅ Final F1 Score
precision = history.history['precision'][-1]
recall = history.history['recall'][-1]
f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
print(f"🔍 Final F1 Score: {f1:.4f}")

# ✅ Save final model
model.save("unet_model.h5")
print("✅ Final model saved as unet_model.h5")

# ✅ Plot Loss Curve
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("training_loss.png")
plt.show()

# ✅ Plot Training Metrics
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Val Accuracy")
plt.plot(history.history['precision'], label="Precision")
plt.plot(history.history['recall'], label="Recall")
plt.title("Training Metrics")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.savefig("training_metrics.png")
plt.show()
