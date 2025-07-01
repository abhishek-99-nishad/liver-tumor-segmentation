import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Paths
DATA_DIR = "preprocessed"
all_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith("_ct.npy")])
print(f"🧠 Found {len(all_files)} CT volumes for training.")

# Load and stack data
X = []
Y = []

for ct_file in all_files:
    mask_file = ct_file.replace("_ct.npy", "_mask.npy")
    ct_path = os.path.join(DATA_DIR, ct_file)
    mask_path = os.path.join(DATA_DIR, mask_file)

    ct = np.load(ct_path)  # shape: (slices, 256, 256)
    mask = np.load(mask_path)

    X.extend(ct)
    Y.extend(mask)

X = np.expand_dims(np.array(X), axis=-1)        # shape: (n, 256, 256, 1)
Y = np.expand_dims(np.array(Y), axis=-1)

print(f"✅ Loaded {X.shape[0]} slices for training.")

# Split data
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Build U-Net model
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

    c1, p1 = encoder_block(inputs, 32)
    c2, p2 = encoder_block(p1, 64)
    c3, p3 = encoder_block(p2, 128)

    bn = conv_block(p3, 256)

    d3 = decoder_block(bn, c3, 128)
    d2 = decoder_block(d3, c2, 64)
    d1 = decoder_block(d2, c1, 32)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(d1)

    model = models.Model(inputs, outputs)
    return model

model = build_unet()
from tensorflow.keras.metrics import Precision, Recall

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', Precision(), Recall()]
)

model.summary()

# Train
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint("unet_model_best.h5", monitor='val_loss', save_best_only=True)

history = model.fit(X_train, Y_train,
                    validation_data=(X_val, Y_val),
                    epochs=10,
                    batch_size=8,
                    callbacks=[checkpoint])


# history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                    # epochs=10, batch_size=8)
# Calculate F1 Score manually
precision = history.history['precision'][-1]
recall = history.history['recall'][-1]
f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
print(f"🔍 Final Training F1 Score: {f1_score:.4f}")


# Save model
model.save("unet_model.h5")
print("✅ Model saved as unet_model.h5")

# Plot training loss
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.legend()
plt.title("U-Net Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("training_loss.png")
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.plot(history.history['precision'], label='Precision')
plt.plot(history.history['recall'], label='Recall')
plt.title("U-Net Training Metrics")
plt.xlabel("Epochs")
plt.ylabel("Metric Value")
plt.legend()
plt.savefig("training_metrics.png")
plt.show()


