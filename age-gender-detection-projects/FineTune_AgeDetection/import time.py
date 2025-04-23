import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ─── CONFIG ────────────────────────────────────────────────────────────────
IMAGE_DIR   = "/Users/vandanakashyap/Downloads/Finetunemodel/UTKFace"  # Change to your UTKFace path
IMAGE_SIZE  = 160       # Smaller input size
BATCH_SIZE  = 32
EPOCHS      = 10
MODEL_OUT   = "age_detection_model_fast.keras"

# ─── OPTIONAL: MIXED PRECISION ──────────────────────────────────────────────
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# ─── 1) GATHER PATHS & AGES ─────────────────────────────────────────────────
records = []
for fname in os.listdir(IMAGE_DIR):
    parts = fname.split('_')
    if not parts or not parts[0].isdigit(): continue
    records.append((os.path.join(IMAGE_DIR, fname), int(parts[0])))

df = pd.DataFrame(records, columns=['path','age'])
print(f"Total images: {len(df)}")

# ─── 2) SPLIT ────────────────────────────────────────────────────────────────
p_train, p_val, y_train, y_val = train_test_split(
    df['path'].tolist(), df['age'].tolist(),
    test_size=0.2, random_state=42
)
print(f"Train: {len(p_train)}, Val: {len(p_val)}")

# ─── 3) DATA PIPELINE WITH CACHE ─────────────────────────────────────────────
def load_and_preprocess(path, age):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE]) / 255.0
    return img, tf.cast(age, tf.float32)

train_ds = (
    tf.data.Dataset.from_tensor_slices((p_train, y_train))
    .shuffle(buffer_size=5000)
    .map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()                                # <-- cache in RAM after first epoch
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((p_val, y_val))
    .map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()                                # cache validation too
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# ─── 4) BUILD A LIGHTER MODEL ───────────────────────────────────────────────
base = tf.keras.applications.MobileNetV2(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    alpha=0.35,            # Smaller network
    include_top=False,
    pooling='avg',
    weights='imagenet'
)
base.trainable = False

inputs = tf.keras.Input([IMAGE_SIZE, IMAGE_SIZE, 3])
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base(x, training=False)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
# For mixed precision, output in float32
outputs = tf.keras.layers.Dense(1, dtype='float32', name='age')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)
model.summary()

# ─── 5) TRAIN ───────────────────────────────────────────────────────────────
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ─── 6) EVALUATE & SAVE ────────────────────────────────────────────────────
val_loss, val_mae = model.evaluate(val_ds)
print(f"\nValidation MAE: {val_mae:.2f} years")

model.save(MODEL_OUT)
print(f"Fast model saved to {MODEL_OUT}")

# ─── 7) PLOT RESULTS ───────────────────────────────────────────────────────
import matplotlib.pyplot as plt

# Plot training & validation MAE
plt.figure()
plt.plot(history.history['mae'], label='train_mae')
plt.plot(history.history['val_mae'], label='val_mae')
plt.title('MAE over epochs')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.show()

# Plot training & validation loss (MSE)
plt.figure()
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()

# ─── 8) FINE-TUNE THE MODEL ────────────────────────────────────────────────
# Unfreeze the last 20% of layers
num_layers = len(base.layers)
for layer in base.layers[int(num_layers * 0.8):]:
    layer.trainable = True

# Recompile with a lower LR
from tensorflow.keras.optimizers import Adam
model.compile(
    optimizer=Adam(1e-5),  # Much lower LR for fine-tuning
    loss='mse',
    metrics=['mae']
)

# Continue training (fewer epochs)
fine_tune_epochs = 5
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=fine_tune_epochs
)

model.save('age_detection_model_finetuned.keras')
