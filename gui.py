import numpy as np
import librosa
import sounddevice as sd
import joblib
import tensorflow as tf
import streamlit as st
import io
import cv2
import threading
import queue
import time

# ====== Streamlit Page Config (FIRST Streamlit command) ======
st.set_page_config(page_title="Emotion Recognition", layout="wide")

try:
    from deepface import DeepFace
    deepface_available = True
except ImportError:
    deepface_available = False

# ====== CONSTANTS ======
SAMPLE_RATE = 16000
N_MFCC = 64
N_MELS = 64
MAX_PAD_LEN = 200

# ====== Load model and label encoder ======
@st.cache_resource
def load_model_and_encoder():
    model = tf.keras.models.load_model("models/best_emotion_model.h5")
    label_encoder = joblib.load("models/label_encoder.pkl")
    return model, label_encoder

model, label_encoder = load_model_and_encoder()

# ====== Feature extraction ======
def extract_features_from_audio_np(audio_np, sr=SAMPLE_RATE, max_pad_len=MAX_PAD_LEN, n_mfcc=N_MFCC, n_mels=N_MELS):
    try:
        mfccs = librosa.feature.mfcc(y=audio_np, sr=sr, n_mfcc=n_mfcc)
        mel_spec = librosa.feature.melspectrogram(y=audio_np, sr=sr, n_mels=n_mels)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        mfccs = librosa.util.normalize(mfccs)
        mel_spec = librosa.util.normalize(mel_spec)
        combined_features = np.vstack((mfccs, mel_spec))
        if combined_features.shape[1] < max_pad_len:
            pad_width = max_pad_len - combined_features.shape[1]
            combined_features = np.pad(combined_features, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            combined_features = combined_features[:, :max_pad_len]
        return combined_features.T[..., np.newaxis]
    except Exception as e:
        st.error(f"❌ Error extracting features: {e}")
        return None

def extract_features_from_filelike(file_obj, sr=SAMPLE_RATE, max_pad_len=MAX_PAD_LEN, n_mfcc=N_MFCC, n_mels=N_MELS):
    try:
        audio_np, _ = librosa.load(file_obj, sr=sr)
        return extract_features_from_audio_np(audio_np, sr=sr, max_pad_len=max_pad_len, n_mfcc=n_mfcc, n_mels=n_mels)
    except Exception as e:
        st.error(f"❌ Error extracting features: {e}")
        return None

# ====== Record audio helper ======
def record_audio(duration=3, fs=SAMPLE_RATE):
    st.info("🎤 Recording for 3 seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    return audio.flatten()

# ====== Convert audio to wav for st.audio ======
def audio_np_to_wav_bytes(audio_np, fs=SAMPLE_RATE):
    import soundfile as sf
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        sf.write(tmp.name, audio_np, fs)
        tmp.seek(0)
        return tmp.read()

# ====== Emoji mapping ======
emotion_emoji = {
    "angry": "😠", "disgust": "🤢", "fearful": "😨", "happy": "😄",
    "neutral": "😐", "sad": "😢", "surprised": "😲", "calm": "😌"
}

# ====== Live microphone emotion detection ======
def live_voice_stream(q, duration=2, sr=SAMPLE_RATE):
    def callback(indata, frames, time_, status):
        if status:
            st.warning(f"Audio status: {status}")
        q.put(indata.copy())

    with sd.InputStream(samplerate=sr, channels=1, callback=callback, blocksize=int(duration * sr)):
        while st.session_state.run_mic:
            time.sleep(0.1)

def run_live_emotion_detection():
    q = queue.Queue()
    threading.Thread(target=live_voice_stream, args=(q,), daemon=True).start()
    st.info("🎤 Live microphone is ON... Speak now!")

    while st.session_state.run_mic:
        try:
            audio = q.get(timeout=5).flatten()
            features = extract_features_from_audio_np(audio)
            if features is not None:
                pred = model.predict(np.expand_dims(features, 0))
                emotion = label_encoder.inverse_transform([np.argmax(pred)])[0]
                emoji = emotion_emoji.get(emotion, "")
                st.success(f"**Live Detected Emotion:** {emotion.capitalize()} {emoji}")
        except queue.Empty:
            st.warning("🎙️ No input detected.")
            break

# ====== Sidebar ======
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/theatre-mask.png", width=80)
    st.markdown("""
    # 🎭 Emotion Recognition
    Welcome!  
    This app detects **emotions** from your **speech** and **face** using AI (Deep Learning).
    ---
    **Instructions:**
    - Use the left section to upload or record speech.
    - Use the right section for live facial emotion detection (webcam).
    - Click the buttons to start/stop live detection.
    - For best results: speak clearly & ensure good lighting!
    ---
    [GitHub Repo](https://github.com/demiko21/SpeechEmotionRecognition)
    """)
    st.markdown("---")
    st.caption("Made with ❤️ and Streamlit")

# ====== Main Interface ======
st.markdown(
    """
    <h1 style='text-align: center; background: linear-gradient(90deg, #ffecd2 0%, #fcb69f 100%); border-radius: 10px; color: #333; padding: 0.5em 0em 0.5em 0em; margin-bottom: 0px;'>
        🎭 Real-Time Emotion Recognition
    </h1>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    "<div style='text-align:center; margin-bottom:20px; font-size:18px;'>"
    "Detect <b>emotions</b> from <b>speech</b> and <b>facial expressions</b> using Artificial Intelligence.<br>"
    "<span style='color:#666;'>Bachelor's Diploma Project &mdash; 2025</span>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("---")

col1, col2 = st.columns(2)

# ====== Audio-based Emotion Recognition ======
with col1:
    st.header("🔊 Speech Emotion Recognition")
    st.write("Upload a `.wav` file or record your voice:")

    uploaded_file = st.file_uploader("Choose a .wav audio file", type=["wav"], label_visibility="collapsed")
    if uploaded_file is not None:
        st.info("🔎 Analyzing uploaded audio...")
        audio_bytes = uploaded_file.read()
        features = extract_features_from_filelike(io.BytesIO(audio_bytes))
        if features is not None:
            pred = model.predict(np.expand_dims(features, 0))
            emotion = label_encoder.inverse_transform([np.argmax(pred)])[0]
            emoji = emotion_emoji.get(emotion, "")
            st.success(f"**Detected Emotion:** <span style='font-size:1.5em'>{emotion.capitalize()} {emoji}</span>", unsafe_allow_html=True)

    col_rec, col_live = st.columns([1,1])
    with col_rec:
        if st.button("🎙️ Record & Analyze Voice"):
            audio = record_audio()
            wav_bytes = audio_np_to_wav_bytes(audio)
            st.audio(wav_bytes, format="audio/wav", sample_rate=SAMPLE_RATE)
            features = extract_features_from_audio_np(audio)
            if features is not None:
                pred = model.predict(np.expand_dims(features, 0))
                emotion = label_encoder.inverse_transform([np.argmax(pred)])[0]
                emoji = emotion_emoji.get(emotion, "")
                st.success(f"**Detected Emotion:** <span style='font-size:1.5em'>{emotion.capitalize()} {emoji}</span>", unsafe_allow_html=True)

    with col_live:
        # Live Microphone Detection
        if "run_mic" not in st.session_state:
            st.session_state.run_mic = False

        mic_status = "🟢 ON" if st.session_state.run_mic else "🔴 OFF"
        st.markdown(f"**Live Mic:** {mic_status}")

        if st.button("▶️ Start Live Mic Detection"):
            st.session_state.run_mic = True
            threading.Thread(target=run_live_emotion_detection, daemon=True).start()
        if st.button("⏹ Stop Live Mic Detection"):
            st.session_state.run_mic = False

    st.markdown("---")

# ====== Face-based Emotion Recognition ======
with col2:
    st.header("📷 Live Facial Emotion Recognition")
    st.write("Turn on your webcam and click start.")

    if not deepface_available:
        st.warning("DeepFace is not installed. Run `pip install deepface`.")
    else:
        if "run_live" not in st.session_state:
            st.session_state.run_live = False
        FRAME_WINDOW = st.empty()
        detected_emotion = st.empty()

        col_start, col_stop = st.columns(2)
        with col_start:
            if st.button("▶️ Start Webcam Detection"):
                st.session_state.run_live = True
        with col_stop:
            if st.button("⏹ Stop Webcam Detection"):
                st.session_state.run_live = False

        if st.session_state.run_live:
            cap = cv2.VideoCapture(0)
            while st.session_state.run_live:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Camera error!")
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    result = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
                    if isinstance(result, list):
                        emotion_face = result[0]['dominant_emotion']
                    else:
                        emotion_face = result['dominant_emotion']
                    emoji_face = emotion_emoji.get(emotion_face.lower(), "")
                    text = f"{emotion_face.capitalize()} {emoji_face}"
                    cv2.putText(rgb_frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    detected_emotion.success(f"**Facial Emotion:** <span style='font-size:1.5em'>{emotion_face.capitalize()} {emoji_face}</span>", unsafe_allow_html=True)
                except Exception:
                    cv2.putText(rgb_frame, "Face Not Detected", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
                    detected_emotion.warning("No face detected.")
                FRAME_WINDOW.image(rgb_frame, channels="RGB")
            cap.release()
            FRAME_WINDOW.empty()
            detected_emotion.empty()

st.markdown("---")
st.caption("© 2025 demiko21 | [GitHub Repo](https://github.com/demiko21/SpeechEmotionRecognition)")
