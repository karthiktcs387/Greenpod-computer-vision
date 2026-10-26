import os
import time
import joblib
import cv2
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt


# Dataset paths
TRAIN_DIR = "train"
TEST_DIR = "test"

OUTPUT_DIR = "outputs"
MODEL_DIR = "models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def extract_features(image_path):
    """Extract color and texture-related features from an image."""

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    image = cv2.resize(image, (128, 128))

    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    features = []

    # HSV color histograms
    for channel in range(3):
        histogram = cv2.calcHist(
            [hsv],
            [channel],
            None,
            [32],
            [0, 256]
        )

        histogram = cv2.normalize(
            histogram,
            histogram
        ).flatten()

        features.extend(histogram)

    # Mean and standard deviation of each HSV channel
    for channel in range(3):
        features.append(float(np.mean(hsv[:, :, channel])))
        features.append(float(np.std(hsv[:, :, channel])))

    return np.array(features, dtype=np.float32)


def load_dataset(folder):
    """Load images and labels from class subfolders."""

    features = []
    labels = []

    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Dataset folder not found: {folder}")

    classes = sorted(
        name for name in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, name))
    )

    if not classes:
        raise ValueError(f"No class folders found in {folder}")

    for class_name in classes:
        class_dir = os.path.join(folder, class_name)

        for filename in sorted(os.listdir(class_dir)):
            if not filename.lower().endswith(IMAGE_EXTENSIONS):
                continue

            image_path = os.path.join(class_dir, filename)

            try:
                feature_vector = extract_features(image_path)
                features.append(feature_vector)
                labels.append(class_name)

            except Exception as error:
                print(f"Skipping {image_path}: {error}")

    if not features:
        raise ValueError(f"No readable images found in {folder}")

    return np.array(features), np.array(labels)


def main():
    print("Loading training images...")
    X_train, y_train = load_dataset(TRAIN_DIR)

    print("Loading testing images...")
    X_test, y_test = load_dataset(TEST_DIR)

    print("\nTraining samples:", len(y_train))
    print("Testing samples:", len(y_test))
    print("Feature count:", X_train.shape[1])

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining Random Forest...")
    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    print(f"Training time: {training_time:.2f} seconds")

    print("\nEvaluating model...")
    start_time = time.time()

    predictions = model.predict(X_test)

    inference_time = time.time() - start_time

    accuracy = accuracy_score(y_test, predictions)

    print(f"\nTest accuracy: {accuracy:.4f}")
    print(f"Test accuracy percentage: {accuracy * 100:.2f}%")
    print(f"Total prediction time: {inference_time:.4f} seconds")

    print("\nClassification Report:")
    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    print(report)

    with open(
        os.path.join(OUTPUT_DIR, "model1_report.txt"),
        "w",
        encoding="utf-8"
    ) as file:
        file.write(report)

    # Confusion matrix
    class_names = model.classes_

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=class_names
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(cmap="Blues", xticks_rotation=30)
    plt.title("Random Forest - Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "model1_confusion_matrix.png"),
        dpi=150
    )

    plt.show()

    # Save the trained model
    joblib.dump(
        model,
        os.path.join(MODEL_DIR, "random_forest.joblib")
    )

    print("\nModel saved to models/random_forest.joblib")
    print("Report saved to outputs/model1_report.txt")


if __name__ == "__main__":
    main()