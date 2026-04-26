from flask import Flask, render_template, request
import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = load_model("models/cnn_lstm_model.h5")

classes = ["Asthma", "Bronchiectasis", "COPD", "Healthy", "Pneumonia"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":

        file = request.files["audio"]
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Load audio
        y, sr = librosa.load(filepath, sr=16000)

        # ===============================
        # 🔥 CNN INPUT (Spectrogram)
        # ===============================
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_db = librosa.power_to_db(mel)

        mel_db = np.resize(mel_db, (128, 128))
        mel_db = np.stack([mel_db]*3, axis=-1)
        mel_db = np.expand_dims(mel_db, axis=0)

        # Prediction
        pred = model.predict(mel_db)
        class_index = np.argmax(pred)
        confidence = round(np.max(pred) * 100, 2)

        prediction = classes[class_index]

        # ===============================
        # 🔥 DYNAMIC FEATURE EXTRACTION
        # ===============================
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfcc, axis=1)

        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))

        feature_values = list(mfcc_mean) + [spectral_contrast, zcr]

        feature_names = [f"MFCC_{i}" for i in range(20)] + ["Spectral Contrast", "Zero Crossing Rate"]

        top_indices = np.argsort(feature_values)[-4:]

        features = [feature_names[i] for i in top_indices]

        return render_template(
            "predict.html",
            result=True,
            filename=file.filename,
            prediction=prediction,
            confidence=confidence,
            features=features
        )

    return render_template("predict.html", result=False)

if __name__ == "__main__":
    app.run(debug=True)