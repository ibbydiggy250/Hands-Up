import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# ============================================================
# Configuration
# ============================================================
DATA_DIR = os.path.join("data", "samples")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# 1️⃣ Load all recorded gesture samples
# ============================================================
X, y = [], []
SEQ_LEN = 30          # number of frames per sequence
FEATURE_LEN = 126     # 21 points × 3 coords × 2 hands

print("[INFO] Loading gesture data from:", DATA_DIR)

for file in os.listdir(DATA_DIR):
    if file.endswith(".npz"):
        data = np.load(os.path.join(DATA_DIR, file))
        x = data["x"]

        # --- normalize sequence length ---
        if len(x) < SEQ_LEN:
            pad_width = ((0, SEQ_LEN - len(x)), (0, 0))
            x = np.pad(x, pad_width, mode='constant')
        elif len(x) > SEQ_LEN:
            x = x[:SEQ_LEN]

        # --- normalize feature dimension ---
        if x.shape[1] < FEATURE_LEN:
            pad_width = ((0, 0), (0, FEATURE_LEN - x.shape[1]))
            x = np.pad(x, pad_width, mode='constant')
        elif x.shape[1] > FEATURE_LEN:
            x = x[:, :FEATURE_LEN]

        X.append(x)
        y.append(str(data["y"]))

X = np.array(X)
y = np.array(y)

print(f"[INFO] Loaded {len(X)} samples.")
print(f"[INFO] Data shape: {X.shape}")

# ============================================================
# 2️⃣ Encode labels numerically
# ============================================================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

with open(os.path.join(MODEL_DIR, "labels.txt"), "w") as f:
    for label in label_encoder.classes_:
        f.write(f"{label}\n")

print("[INFO] Labels:", list(label_encoder.classes_))

# ============================================================
# 3️⃣ Split into training/testing sets
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)
print(f"[INFO] Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ============================================================
# 4️⃣ Build the LSTM model
# ============================================================
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2]))),
    Dropout(0.4),
    Bidirectional(LSTM(64)),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ============================================================
# 5️⃣ Train the model
# ============================================================
checkpoint_path = os.path.join(MODEL_DIR, "asl_model.h5")
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1)

print("[INFO] Starting training...\n")

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=40,
    batch_size=8,
    callbacks=[checkpoint],
    verbose=1
)

print("\n[✅] Training complete! Best model saved to:", checkpoint_path)

# ============================================================
# 6️⃣ Evaluate performance
# ============================================================
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"[INFO] Final Test Accuracy: {acc * 100:.2f}%")

model.save(checkpoint_path)
print("[INFO] Model and labels saved successfully!")
