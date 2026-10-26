import os
import sys
import cv2
import joblib
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


IMG_SIZE = (160, 160)


def extract_features(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    image = cv2.resize(image, (128, 128))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    features = []

    for channel in range(3):
        histogram = cv2.calcHist(
            [hsv], [channel], None, [32], [0, 256]
        )
        histogram = cv2.normalize(histogram, histogram).flatten()
        features.extend(histogram)

    for channel in range(3):
        features.append(float(np.mean(hsv[:, :, channel])))
        features.append(float(np.std(hsv[:, :, channel])))

    return np.array(features, dtype=np.float32)


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        return

    image_path = sys.argv[1]

    if not os.path.isfile(image_path):
        print(f"Image not found: {image_path}")
        return

    print("\nSelect a model:")
    print("1. Random Forest")
    print("2. MobileNetV2")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        model = joblib.load("models/random_forest.joblib")
        features = extract_features(image_path).reshape(1, -1)

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = float(np.max(probabilities))

    elif choice == "2":
        model = tf.keras.models.load_model("models/mobilenetv2.keras")

        image = cv2.imread(image_path)

        if image is None:
            print("Unable to read image.")
            return

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMG_SIZE)
        image = preprocess_input(image.astype(np.float32))
        image = np.expand_dims(image, axis=0)

        probabilities = model.predict(image, verbose=0)[0]
        predicted_index = int(np.argmax(probabilities))

        # Obtain class names from the training directory.
        class_names = sorted(
            name for name in os.listdir("train")
            if os.path.isdir(os.path.join("train", name))
        )

        prediction = class_names[predicted_index]
        confidence = float(probabilities[predicted_index])

    else:
        print("Invalid selection.")
        return

    print("\n========== PREDICTION RESULT ==========")
    print("Image:", image_path)
    print("Predicted class:", prediction)
    print(f"Model confidence: {confidence * 100:.2f}%")


if __name__ == "__main__":
    main()
