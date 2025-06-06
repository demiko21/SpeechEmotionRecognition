import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import joblib
import queue
from collections import deque, Counter
from colorama import Fore, Style, init

# ====== Initialize colorama for colored output ======
init(autoreset=True)

# ====== Constants ======
SR = 16000
DURATION = 2  # seconds
N_MFCC = 64
N_MELS = 64
MAX_PAD_LEN = 200

# ====== Load model and label encoder (using updated paths) ======
model = tf.keras.models.load_model("models/best_emotion_model.h5")
label_encoder = joblib.load("models/label_encoder.pkl")

# ====== Color map for terminal output ======
color_map = {
    "angry": Fore.RED,
    "disgust": Fore.CYAN,
    "fearful": Fore.MAGENTA,
    "happy": Fore.GREEN,
    "neutral": Fore.YELLOW,
    "sad": Fore.BLUE,
    "surprised": Fore.LIGHTWHITE_EX,
    "calm": Fore.LIGHTGREEN_EX
}

# ====== Extract audio features ======
def extract_features(y, sr=SR, max_pad_len=MAX_PAD_LEN):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    mfccs = librosa.util.normalize(mfccs)
    mel_spec = librosa.util.normalize(mel_spec)
    combined_features = np.vstack((mfccs, mel_spec))
    if combined_features.shape[1] < max_pad_len:
        pad_width = max_pad_len - combined_features.shape[1]
        combined_features = np.pad(combined_features, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        combined_features = combined_features[:, :max_pad_len]
    return combined_features.T[np.newaxis, ..., np.newaxis]

# ====== Live audio emotion detection ======
def live_detect():
    q = queue.Queue()
    history = deque(maxlen=5)

    def callback(indata, frames, time, status):
        if status:
            print(Fore.RED + f"⚠️ Stream status: {status}")
        q.put(indata.copy())

    print(Fore.CYAN + "🎤 Speak to the microphone for real-time emotion detection... (Press Ctrl+C to stop)\n")

    try:
        with sd.InputStream(samplerate=SR, channels=1, callback=callback, blocksize=SR * DURATION):
            while True:
                audio = q.get()
                audio = audio.flatten()

                # Silence detection
                if np.max(np.abs(audio)) < 0.01:
                    print(Fore.LIGHTBLACK_EX + "🔇 No speech detected... Speak louder or closer to mic.")
                    continue

                features = extract_features(audio)
                pred = model.predict(features, verbose=0)
                probs = tf.nn.softmax(pred[0]).numpy()

                top_indices = np.argsort(probs)[-3:][::-1]
                print(Fore.WHITE + "-" * 40)
                for i in top_indices:
                    emotion = label_encoder.inverse_transform([i])[0]
                    confidence = probs[i]
                    color = color_map.get(emotion, Fore.WHITE)
                    print(f"{color}{emotion.capitalize():<12}: {confidence:.2%}")

                # Smooth output over recent predictions
                top_emotion = label_encoder.inverse_transform([np.argmax(pred)])[0]
                history.append(top_emotion)
                smoothed = Counter(history).most_common(1)[0][0]
                print(Fore.LIGHTYELLOW_EX + f"\n🧠 Smoothed Emotion: {smoothed.capitalize()}")
                print("-" * 40)

    except KeyboardInterrupt:
        print(Fore.LIGHTRED_EX + "\n🛑 Stopped live detection. Goodbye!")

if __name__ == "__main__":
    live_detect()
