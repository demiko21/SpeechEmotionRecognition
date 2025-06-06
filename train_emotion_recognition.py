import os
import glob
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout,
    BatchNormalization, Input, Reshape, MultiHeadAttention,
    LayerNormalization, Add, Bidirectional
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

try:
    import soundfile as sf
except ImportError:
    print("⚠️ soundfile is not installed. Install it with `pip install soundfile`.\n")

# ========== Configuration ==========
DATASET_PATH = "datasett"
OUTPUT_DIR = "models"
SR = 16000
N_MFCC = 64
N_MELS = 64
MAX_PAD_LEN = 200
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 10
AUGMENT = False
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)
warnings.filterwarnings("ignore")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== Data Augmentation ==========
def augment_audio(y, sr):
    aug_type = np.random.choice(['none', 'noise', 'pitch', 'stretch'])
    if aug_type == 'noise':
        noise_amp = 0.005 * np.random.uniform() * np.amax(y)
        y = y + noise_amp * np.random.normal(size=y.shape[0])
    elif aug_type == 'pitch':
        y = librosa.effects.pitch_shift(y, sr, n_steps=np.random.randint(-2, 3))
    elif aug_type == 'stretch':
        rate = np.random.uniform(0.8, 1.2)
        y = librosa.effects.time_stretch(y, rate)
        if len(y) > MAX_PAD_LEN * 100:
            y = y[:MAX_PAD_LEN * 100]
    return y

# ========== Feature Extraction ==========
def extract_features(file_path, max_pad_len=MAX_PAD_LEN, augment=AUGMENT):
    try:
        y, sr = librosa.load(file_path, sr=SR, res_type='kaiser_fast')
        if augment:
            y = augment_audio(y, sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        mfccs = librosa.util.normalize(mfccs)
        mel_spec = librosa.util.normalize(mel_spec)
        combined = np.vstack((mfccs, mel_spec))
        if combined.shape[1] < max_pad_len:
            pad_width = max_pad_len - combined.shape[1]
            combined = np.pad(combined, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            combined = combined[:, :max_pad_len]
        return combined.T[..., np.newaxis]
    except Exception as e:
        print(f"❌ Error extracting features from {file_path}: {e}")
        return None

# ========== Dataset Loading ==========
def load_data():
    emotions_map = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }
    tess_emotions = {
        "angry": "angry", "disgust": "disgust", "fear": "fearful",
        "happy": "happy", "neutral": "neutral", "ps": "surprised",
        "sad": "sad"
    }
    x, y = [], []

    # RAVDESS
    print("🔎 Loading RAVDESS dataset...")
    ravdess_files = glob.glob(os.path.join(DATASET_PATH, "Actor_*", "*.wav"))
    for file in tqdm(ravdess_files, desc="RAVDESS"):
        emotion_code = os.path.basename(file).split("-")[2]
        emotion = emotions_map.get(emotion_code)
        if emotion:
            features = extract_features(file)
            if features is not None:
                x.append(features)
                y.append(emotion)

    # TESS
    print("🔎 Loading TESS dataset...")
    tess_files = glob.glob(os.path.join(DATASET_PATH, "TESS", "*", "*.wav"))
    for file in tqdm(tess_files, desc="TESS"):
        folder = os.path.basename(os.path.dirname(file)).lower()
        emotion = next((v for k, v in tess_emotions.items() if k in folder), None)
        if emotion:
            features = extract_features(file)
            if features is not None:
                x.append(features)
                y.append(emotion)

    return np.array(x), np.array(y)

# ========== Transformer Block ==========
def transformer_block(inputs, num_heads=4, ff_dim=128):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(inputs, inputs)
    attn_output = Add()([attn_output, inputs])
    attn_output = LayerNormalization(epsilon=1e-6)(attn_output)
    ffn = Dense(ff_dim, activation="relu")(attn_output)
    ffn = Dense(inputs.shape[-1])(ffn)
    ffn_output = Add()([ffn, attn_output])
    return LayerNormalization(epsilon=1e-6)(ffn_output)

# ========== Model Architecture ==========
def create_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    s1, s2, s3 = x.shape[1], x.shape[2], x.shape[3]
    x = Reshape((s1, s2 * s3))(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = LSTM(128, return_sequences=True)(x)
    x = transformer_block(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# ========== Plotting ==========
def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Train Acc')
    plt.plot(history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.title("Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_plot.png"))
    plt.close()

def plot_confusion(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()

# ========== Training Script ==========
if __name__ == "__main__":
    print("🔄 Loading data...")
    x, y = load_data()
    if len(x) == 0:
        print("❌ No data loaded. Check dataset path and formats.")
        exit(1)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    joblib.dump(label_encoder, os.path.join(OUTPUT_DIR, "label_encoder.pkl"))

    x_train, x_test, y_train, y_test = train_test_split(
        x, y_encoded, test_size=0.2, stratify=y_encoded, random_state=SEED
    )

    print("🔧 Building model...")
    model = create_model(x_train.shape[1:], len(np.unique(y_encoded)))
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
        ModelCheckpoint(os.path.join(OUTPUT_DIR, "best_emotion_model.h5"), save_best_only=True)
    ]

    print("🚀 Training...")
    history = model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        callbacks=callbacks
    )

    print("💾 Saving final model...")
    model.save(os.path.join(OUTPUT_DIR, "final_emotion_model.h5"))
    joblib.dump(history.history, os.path.join(OUTPUT_DIR, "training_history.pkl"))

    print("📈 Plotting...")
    plot_history(history.history)

    print("📊 Evaluating...")
    y_pred = np.argmax(model.predict(x_test), axis=1)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    print("🔹 Classification Report:\n", report)
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    plot_confusion(y_test, y_pred, label_encoder.classes_)
    print("✅ Done! All artifacts saved in 'models/' folder.")
