import os
import io

import cv2
import joblib
import numpy as np

from PIL import Image
from flask import Flask, request, render_template_string


# --------------------------------------------------
# Flask application configuration
# --------------------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR, "models", "random_forest.joblib"
)

# Update these labels if your training script uses
# a different class order.
CLASS_NAMES = ["blackspot", "canker", "fresh", "grenning"]


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model = None

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("Random Forest model loaded successfully.")
    except Exception as exc:
        print(f"Error loading model: {exc}")
else:
    print(f"Model file not found: {MODEL_PATH}")


# --------------------------------------------------
# Image feature extraction
# --------------------------------------------------

def extract_features(image_bytes):
    """Extract HSV histograms and channel statistics."""

    with Image.open(io.BytesIO(image_bytes)) as image:
        rgb_image = np.array(image.convert("RGB"))

    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    hsv_image = cv2.resize(hsv_image, (128, 128))

    features = []

    # Extract a 32-bin histogram from each HSV channel.
    for channel in range(3):
        histogram = cv2.calcHist(
            [hsv_image],
            [channel],
            None,
            [32],
            [0, 256],
        )

        histogram = cv2.normalize(
            histogram,
            histogram,
        ).flatten()

        features.extend(histogram.tolist())

    # Extract mean and standard deviation for each channel.
    for channel in range(3):
        channel_data = hsv_image[:, :, channel]

        features.append(float(np.mean(channel_data)))
        features.append(float(np.std(channel_data)))

    return np.asarray(features, dtype=np.float32).reshape(1, -1)


# --------------------------------------------------
# HTML interface
# --------------------------------------------------

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>Orange Condition Classifier</title>

    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f4f7fb;
            margin: 0;
            padding: 30px 15px;
            color: #222;
        }

        .container {
            max-width: 650px;
            margin: 35px auto;
            background: white;
            padding: 30px;
            border-radius: 14px;
            box-shadow: 0 5px 22px rgba(0, 0, 0, 0.08);
        }

        h1 {
            text-align: center;
            color: #d97706;
        }

        .description {
            text-align: center;
            color: #555;
            line-height: 1.6;
        }

        form {
            margin-top: 25px;
            text-align: center;
        }

        input[type="file"] {
            display: block;
            width: 100%;
            margin: 20px 0;
        }

        button {
            background: #e8790c;
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
        }

        button:hover {
            background: #c65f05;
        }

        .result {
            margin-top: 25px;
            padding: 18px;
            background: #ecfdf5;
            border-radius: 8px;
            text-align: center;
        }

        .error {
            margin-top: 20px;
            padding: 15px;
            background: #fef2f2;
            color: #b91c1c;
            border-radius: 8px;
        }

        .preview {
            display: block;
            max-width: 100%;
            max-height: 300px;
            margin: 20px auto;
            border-radius: 8px;
        }

        .footer {
            text-align: center;
            color: #777;
            margin-top: 25px;
            font-size: 13px;
        }
    </style>
</head>

<body>
    <div class="container">
        <h1>Orange Condition Classifier</h1>

        <p class="description">
            Upload an orange image to classify its condition
            using a machine learning model.
        </p>

        <form method="POST" enctype="multipart/form-data">
            <input
                type="file"
                name="image"
                accept=".jpg,.jpeg,.png,.webp"
                required
            >

            <button type="submit">Classify Image</button>
        </form>

        {% if result %}
            <div class="result">
                <h2>Prediction: {{ result }}</h2>
                {% if confidence is not none %}
                    <p>Model confidence: {{ confidence }}%</p>
                {% endif %}
                <p>This is a model prediction, not a guarantee
                of the fruit's actual condition.</p>
            </div>
        {% endif %}

        {% if error %}
            <div class="error">
                <strong>{{ error }}</strong>
            </div>
        {% endif %}

        {% if image_data %}
            <img
                class="preview"
                src="data:image/jpeg;base64,{{ image_data }}"
                alt="Uploaded orange"
            >
        {% endif %}

        <div class="footer">
            Orange Image Classification | Machine Learning Project
        </div>
    </div>
</body>
</html>
"""


# --------------------------------------------------
# Main page and prediction route
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    error = None
    image_data = None

    if request.method == "POST":
        uploaded_file = request.files.get("image")

        if uploaded_file is None or uploaded_file.filename == "":
            error = "Please select an image."
        elif model is None:
            error = (
                "The trained model is not available. "
                "Please check the model file and deployment setup."
            )
        else:
            try:
                import base64

                image_bytes = uploaded_file.read()

                # Validate that the uploaded file is an image.
                with Image.open(io.BytesIO(image_bytes)) as image:
                    image.verify()

                features = extract_features(image_bytes)

                prediction = model.predict(features)[0]
                result = str(prediction)

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(features)[0]
                    confidence = round(
                        float(np.max(probabilities)) * 100,
                        2,
                    )

                image_data = base64.b64encode(
                    image_bytes
                ).decode("utf-8")

            except Exception as exc:
                error = f"Unable to classify this image: {exc}"

    return render_template_string(
        HTML,
        result=result,
        confidence=confidence,
        error=error,
        image_data=image_data,
    )


# --------------------------------------------------
# Health check route for Render
# --------------------------------------------------

@app.route("/health")
def health():
    status_code = 200 if model is not None else 503

    return {
        "status": "ok" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
    }, status_code


# --------------------------------------------------
# Run locally
# --------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)