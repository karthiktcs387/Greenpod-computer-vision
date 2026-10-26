import os
import hashlib
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False

# Dataset location
DATASET_DIR = "."


# Supported image formats
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Output directory
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_image_files(folder):
    """Return image paths inside a folder."""
    image_files = []

    if not os.path.isdir(folder):
        print(f"WARNING: Folder not found: {folder}")
        return image_files

    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(IMAGE_EXTENSIONS):
                image_files.append(os.path.join(root, file))

    return image_files


def analyze_dataset():
    records = []
    corrupt_images = []
    image_hashes = {}
    duplicate_images = []

    splits = ["train", "test"]

    for split in splits:
        split_dir = os.path.join(DATASET_DIR, split)

        if not os.path.isdir(split_dir):
            print(f"WARNING: Missing split folder: {split_dir}")
            continue

        classes = sorted(
            name for name in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, name))
        )

        for class_name in classes:
            class_dir = os.path.join(split_dir, class_name)
            image_paths = get_image_files(class_dir)

            print(f"{split} | {class_name}: {len(image_paths)} images")

            for image_path in image_paths:
                try:
                    with Image.open(image_path) as image:
                        image.verify()

                    with Image.open(image_path) as image:
                        image = image.convert("RGB")
                        width, height = image.size

                        if width < 1 or height < 1:
                            raise ValueError("Invalid image dimensions")

                        # Exact duplicate detection
                        with open(image_path, "rb") as file:
                            image_hash = hashlib.sha256(
                                file.read()
                            ).hexdigest()

                        if image_hash in image_hashes:
                            duplicate_images.append({
                                "image": image_path,
                                "duplicate_of": image_hashes[image_hash],
                                "split": split,
                                "class": class_name
                            })
                        else:
                            image_hashes[image_hash] = image_path

                        records.append({
                            "split": split,
                            "class": class_name,
                            "image_path": image_path,
                            "width": width,
                            "height": height,
                            "aspect_ratio": round(width / height, 3),
                            "file_size_kb": round(
                                os.path.getsize(image_path) / 1024, 2
                            )
                        })

                except Exception as error:
                    corrupt_images.append({
                        "image": image_path,
                        "error": str(error)
                    })

    if not records:
        print("\nNo valid images found.")
        print("Check DATASET_DIR and your folder structure.")
        return

    df = pd.DataFrame(records)

    # 1. Dataset summary
    print("\n========== DATASET SUMMARY ==========")
    print("Valid images:", len(df))
    print("Corrupted images:", len(corrupt_images))
    print("Exact duplicate occurrences:", len(duplicate_images))

    print("\n========== CLASS DISTRIBUTION ==========")
    distribution = pd.crosstab(df["split"], df["class"])
    print(distribution)

    print("\n========== IMAGE DIMENSIONS ==========")
    print(df[["width", "height"]].describe())

    print("\n========== IMAGE FILE SIZE (KB) ==========")
    print(df["file_size_kb"].describe())

    # 2. Save detailed image information
    df.to_csv(
        os.path.join(OUTPUT_DIR, "image_inventory.csv"),
        index=False
    )

    distribution.to_csv(
        os.path.join(OUTPUT_DIR, "class_distribution.csv")
    )

    pd.DataFrame(corrupt_images).to_csv(
        os.path.join(OUTPUT_DIR, "corrupt_images.csv"),
        index=False
    )

    pd.DataFrame(duplicate_images).to_csv(
        os.path.join(OUTPUT_DIR, "duplicate_images.csv"),
        index=False
    )

    # 3. Visualize class distribution
    distribution.T.plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.title("Image Distribution by Class and Dataset Split")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "class_distribution.png"),
        dpi=150
    )
    plt.show()

    # 4. Display sample images from each class
    classes = sorted(df["class"].unique())

    fig, axes = plt.subplots(
        len(classes), 4,
        figsize=(12, 3 * len(classes))
    )

    if len(classes) == 1:
        axes = axes.reshape(1, -1)

    for row, class_name in enumerate(classes):
        samples = df[df["class"] == class_name].head(4)

        for col in range(4):
            ax = axes[row, col]
            ax.axis("off")

            if col < len(samples):
                image_path = samples.iloc[col]["image_path"]

                with Image.open(image_path) as image:
                    ax.imshow(image.convert("RGB"))

                ax.set_title(class_name)

    plt.suptitle("Sample Images by Class")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTPUT_DIR, "sample_images.png"),
        dpi=150
    )
    plt.show()

    # 5. Check potential extreme image dimensions
    print("\n========== POTENTIAL IMAGE OUTLIERS ==========")

    for column in ["width", "height", "file_size_kb"]:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = df[
            (df[column] < lower) |
            (df[column] > upper)
        ]

        print(f"{column}: {len(outliers)} potential outliers")

    # 6. Check class imbalance
    print("\n========== CLASS IMBALANCE ==========")

    for split in df["split"].unique():
        split_counts = df[df["split"] == split]["class"].value_counts()

        if len(split_counts) > 1:
            imbalance_ratio = (
                split_counts.max() / split_counts.min()
            )

            print(
                f"{split}: imbalance ratio = "
                f"{imbalance_ratio:.2f}:1"
            )

    print("\nAnalysis completed successfully.")
    print(f"Results saved in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    analyze_dataset()
