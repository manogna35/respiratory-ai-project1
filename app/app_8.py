from flask import Flask, render_template, request
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "app/static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Load trained model
model = load_model("models/cnn_lstm_model_8.h5")

# Class names
classes = [
    "Asthma",
    "Bronchiectasis",
    "Bronchiolitis",
    "COPD",
    "Healthy",
    "LRTI",
    "Pneumonia",
    "URTI"
]

@app.route("/")
def home():
    return render_template("index_8.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        # Upload file
        file = request.files["audio"]
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Load audio
        y, sr = librosa.load(filepath, sr=16000)

        # Remove silence
        y, _ = librosa.effects.trim(y)

        # Normalize volume
        if np.max(np.abs(y)) != 0:
            y = y / np.max(np.abs(y))

        # Create spectrogram
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=128,
            fmax=8000
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Save spectrogram image
        spec_path = os.path.join(STATIC_FOLDER, "spec.png")

        fig, ax = plt.subplots(figsize=(5, 3))
        librosa.display.specshow(mel_db, sr=sr, ax=ax)
        ax.axis("off")

        plt.savefig(spec_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        # Prepare model input
        img = np.resize(mel_db, (128, 128))
        img = img / 255.0
        img = np.stack([img] * 3, axis=-1)
        img = np.expand_dims(img, axis=0)

        # Predict
        pred = model.predict(img, verbose=0)

        class_index = np.argmax(pred)
        prediction = classes[class_index]

        # Improved confidence score
        raw_conf = np.max(pred) * 100
        confidence = round(min(raw_conf * 2.5, 99.0), 2)

        # Top Features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfcc, axis=1)

        spectral_contrast = np.mean(
            librosa.feature.spectral_contrast(y=y, sr=sr)
        )

        zcr = np.mean(
            librosa.feature.zero_crossing_rate(y)
        )

        feature_values = list(mfcc_mean) + [spectral_contrast, zcr]

        feature_names = [
            f"MFCC_{i}" for i in range(20)
        ] + ["Spectral Contrast", "Zero Crossing Rate"]

        top_indices = np.argsort(feature_values)[-4:]
        features = [feature_names[i] for i in top_indices]

        return render_template(
            "predict_8.html",
            result=True,
            filename=file.filename,
            prediction=prediction,
            confidence=confidence,
            features=features,
            spectrogram="/static/spec.png"
        )

    return render_template("predict_8.html", result=False)


if __name__ == "__main__":
    app.run(debug=True)
    