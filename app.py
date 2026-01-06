import streamlit as st
import numpy as np
import librosa
import tensorflow as tf

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = "genre_cnn_model.h5"   
model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------------
# Genres (Adjust if needed)
# -----------------------------
GENRES = [
    "blues","classical","country","disco","hiphop",
    "jazz","metal","pop","reggae","rock"
]

MAX_LEN = 1300   # MUST match your training value


# -----------------------------
# MEL FEATURE EXTRACTION
# -----------------------------
def extract_mel(file_path):
    try:
        y, sr = librosa.load(file_path, duration=30, mono=True)

        # Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Normalize 0–1
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())

        # Pad / Trim to fixed size
        if mel_db.shape[1] < MAX_LEN:
            pad = MAX_LEN - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0,0),(0,pad)))
        else:
            mel_db = mel_db[:, :MAX_LEN]

        return mel_db
    
    except Exception as e:
        print("Error:", e)
        return None


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🎵 Music Genre Recognition (Mel Spectrogram Based)")
st.write("Upload a song and the AI will predict its genre!")

uploaded_file = st.file_uploader("Upload audio file", type=["wav","mp3","ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    # Save file temporarily
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.write("Extracting Mel Spectrogram Features...")

    features = extract_mel("temp.wav")

    if features is not None:
        features = np.array(features)

        # Add channel dimension
        features = features[..., np.newaxis]

        # Add batch dimension
        data = np.expand_dims(features, axis=0)

        st.write("Input Shape Sent to Model:", data.shape)

        prediction = model.predict(data)
        predicted_genre = GENRES[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.success(f"🎯 Predicted Genre: **{predicted_genre.upper()}**")
        st.write(f"Confidence: {confidence:.2f}%")

    else:
        st.error("Feature extraction failed. Try another file.")
