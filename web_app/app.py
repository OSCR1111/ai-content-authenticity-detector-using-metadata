from flask import Flask, render_template, request
import os
import joblib
import pandas as pd
import sys

# Allow imports from metadata_extraction
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "metadata_extraction"))

from extract_metadata import extract_metadata
from feature_engineering import metadata_to_features

app = Flask(__name__)

MODEL_PATH = "results/model.pkl"
UPLOAD_FOLDER = "web_app/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            metadata = extract_metadata(file_path)
            features = metadata_to_features(metadata, label=0)

            df = pd.DataFrame([features])
            df = df.drop(columns=["label"])
            df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

            pred = model.predict(df)[0]
            prediction = "AI Generated" if pred == 1 else "Real Image"

    return f"""
    <h2>AI Content Authenticity Detector</h2>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="image" required>
        <br><br>
        <button type="submit">Check Image</button>
    </form>
    <h3>{prediction if prediction else ""}</h3>
    """

if __name__ == "__main__":
    app.run(debug=True)