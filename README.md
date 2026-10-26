# Greenpod Labs – Computer Vision Classification Project

## 1. Project Overview

This project implements a computer vision pipeline to classify fruit images into four categories using machine learning and deep learning techniques.

The two approaches implemented are:

1. **Random Forest with HSV Color Histograms**
2. **MobileNetV2 using Transfer Learning**

The project covers dataset analysis, image preprocessing, feature extraction, model training, evaluation, and image prediction.

**Important dataset note:** The assignment description refers to avocado ripening-stage classification, but the dataset used in this implementation contains orange condition classes. The intended dataset should be confirmed with Greenpod Labs before final submission.

---

## 2. Project Objectives

* Analyze the image dataset and its class distribution.
* Identify possible data-quality issues and class imbalance.
* Implement two different image-classification approaches.
* Evaluate model performance using classification metrics.
* Compare the models and identify their limitations.
* Predict the category of a new fruit image.
* Document potential improvements for real-world deployment.

---

## 3. Project Structure

```text
greenpod-computer-vision/
│
├── outputs/
│
├── train/
│   ├── blackspot/
│   ├── canker/
│   ├── fresh/
│   └── grenning/
│
├── test/
│   ├── blackspot/
│   ├── canker/
│   ├── fresh/
│   └── grenning/
│
├── dataset_analysis.py
├── train_model1.py
├── train_model2.py
├── predict.py
├── requirements.txt
├── .gitignore
└── README.md
```

**Note:** The `train/` and `test/` folders contain the dataset images. They may be kept locally rather than uploaded to GitHub if the dataset is large or subject to sharing restrictions.

---

## 4. Dataset Description

The dataset contains images organized into four classes:

* **Blackspot**
* **Canker**
* **Fresh**
* **Grenning**

### Dataset Statistics

| Class     | Training Images | Testing Images | Total Images |
| --------- | --------------: | -------------: | -----------: |
| Blackspot |             184 |             22 |          206 |
| Canker    |             179 |             22 |          201 |
| Fresh     |             281 |             33 |          314 |
| Grenning  |             347 |             22 |          369 |
| **Total** |         **991** |         **99** |     **1090** |

### Dataset Analysis

The dataset analysis examined:

* Image counts for each class.
* Class imbalance.
* Image dimensions.
* Image file sizes.
* Potential corrupted images.
* Exact duplicate images using SHA-256 hashes.
* Representative images from each category.

The dataset has more images in the Fresh and Grenning classes than in the Blackspot and Canker classes. This imbalance may affect model performance.

---

## 5. Technologies Used

* Python
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* TensorFlow
* Keras
* MobileNetV2
* Git
* GitHub
* Visual Studio Code

---

## 6. Approach 1: Random Forest with HSV Color Histograms

### Overview

The first approach uses handcrafted color features extracted from images and a Random Forest classifier to predict the fruit category.

### Methodology

1. Load images from the training and testing directories.
2. Convert images into the HSV color space using OpenCV.
3. Extract HSV color histograms.
4. Calculate statistical features, including means and standard deviations.
5. Train a Random Forest classifier.
6. Use balanced class weights to help address class imbalance.
7. Evaluate predictions using accuracy and a confusion matrix.

### Model Configuration

| Parameter           | Value                         |
| ------------------- | ----------------------------- |
| Algorithm           | Random Forest                 |
| Number of Trees     | 200                           |
| Feature Extraction  | HSV Histograms                |
| Additional Features | Means and Standard Deviations |
| Class Weight        | Balanced                      |
| Random State        | 42                            |

### Results

| Metric                | Result |
| --------------------- | -----: |
| Correct Predictions   |     96 |
| Incorrect Predictions |      3 |
| Total Test Images     |     99 |
| Accuracy              | 96.97% |

### Confusion Matrix

Rows represent actual classes, and columns represent predicted classes.

| Actual / Predicted | Blackspot | Canker | Fresh | Grenning |
| ------------------ | --------: | -----: | ----: | -------: |
| Blackspot          |        21 |      1 |     0 |        0 |
| Canker             |         2 |     20 |     0 |        0 |
| Fresh              |         0 |      0 |    33 |        0 |
| Grenning           |         0 |      0 |     0 |       22 |

### Observation

The Random Forest model achieved 96.97% observed accuracy on the available test set.

Most classification errors occurred between Blackspot and Canker. Their color characteristics may overlap, making these categories more difficult to distinguish using color features alone.

---

## 7. Approach 2: MobileNetV2 Transfer Learning

### Overview

The second approach uses MobileNetV2, a convolutional neural network pretrained on ImageNet.

Transfer learning allows a pretrained model to extract useful visual features without training an entire deep neural network from scratch.

### Methodology

1. Load images from the dataset.
2. Resize images to 160 × 160 pixels.
3. Prepare images in batches.
4. Load the pretrained MobileNetV2 architecture.
5. Freeze the pretrained feature-extraction layers.
6. Add a global average pooling layer.
7. Add dropout to help reduce overfitting.
8. Add a dense output layer with softmax activation.
9. Train the classification head.
10. Evaluate the model using classification metrics and a confusion matrix.

### Model Configuration

| Parameter          | Value                            |
| ------------------ | -------------------------------- |
| Architecture       | MobileNetV2                      |
| Pretrained Weights | ImageNet                         |
| Input Size         | 160 × 160                        |
| Batch Size         | 16                               |
| Backbone           | Frozen                           |
| Dropout            | 0.25                             |
| Optimizer          | Adam                             |
| Loss Function      | Sparse Categorical Cross-Entropy |
| Training Control   | Early Stopping                   |

### Results

| Metric                     | Result |
| -------------------------- | -----: |
| Correct Predictions        |     95 |
| Incorrect Predictions      |      4 |
| Total Test Images          |     99 |
| Accuracy                   | 95.96% |
| Approximate Macro F1-score |   0.95 |

### Confusion Matrix

| Actual / Predicted | Blackspot | Canker | Fresh | Grenning |
| ------------------ | --------: | -----: | ----: | -------: |
| Blackspot          |        21 |      1 |     0 |        0 |
| Canker             |         3 |     19 |     0 |        0 |
| Fresh              |         0 |      0 |    33 |        0 |
| Grenning           |         0 |      0 |     0 |       22 |

### Evaluation Limitation

The current MobileNetV2 training script uses the test dataset as validation data for early stopping. This means the test data influenced model selection.

Consequently, the reported accuracy is not a fully independent final test estimate.

For a more reliable evaluation, the training data should be divided into training and validation sets, while the test set should remain untouched until the final evaluation.

---

## 8. Model Comparison

| Feature                    | Random Forest                           | MobileNetV2                                      |
| -------------------------- | --------------------------------------- | ------------------------------------------------ |
| Approach                   | Traditional Machine Learning            | Deep Learning                                    |
| Feature Extraction         | HSV Histograms and Statistics           | Pretrained CNN                                   |
| Training Complexity        | Lower                                   | Higher                                           |
| Computational Requirements | Lower                                   | Higher                                           |
| Observed Accuracy          | 96.97%                                  | 95.96%                                           |
| Main Limitation            | Sensitive to color and lighting changes | Requires more computation and careful validation |

### Model Selection

Random Forest achieved the higher observed accuracy on the available test set and has relatively low computational requirements.

However, the results should be interpreted cautiously because MobileNetV2 used the test dataset for validation during training.

A fair comparison requires retraining the models using a separate validation set and evaluating both on an untouched test set.

---

## 9. Example Prediction

The `predict.py` script allows a user to select a model and classify an image.

### Example Command

```bash
python predict.py "test/fresh/f (1).png"
```

Use an image path that actually exists in your dataset.

The script allows the user to choose between Random Forest and MobileNetV2.

### Example Output

```text
Selected Model: Random Forest
Predicted Class: fresh
Confidence: 100.00%
```

The same example image was also classified as Fresh by MobileNetV2.

**Important:** The displayed confidence is the model's output score for that prediction. A confidence score of 100% does not guarantee that the prediction is correct.

---

## 10. Installation and Setup

### Prerequisites

Install the following before running the project:

* Python
* Visual Studio Code
* Git
* The project dataset

### Step 1: Clone the Repository

Replace `YOUR_USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR_USERNAME/greenpod-computer-vision.git
cd greenpod-computer-vision
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 3: Activate the Virtual Environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

For Windows Command Prompt:

```cmd
venv\Scripts\activate.bat
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure the dataset is available in the expected directory structure before running the scripts.

---

## 11. Running the Project

### Analyze the Dataset

```bash
python dataset_analysis.py
```

This script analyzes the dataset and generates the configured analysis outputs.

### Train the Random Forest Model

```bash
python train_model1.py
```

This script extracts HSV features and trains the Random Forest classifier.

### Train the MobileNetV2 Model

```bash
python train_model2.py
```

This script trains the MobileNetV2 transfer-learning model.

### Predict an Image

```bash
python predict.py "test/fresh/f (1).png"
```

Replace the example image path with the path to an existing image.

**Note:** The prediction script may require trained model files generated by the training scripts. Follow the file paths configured in the actual Python scripts.

---

## 12. Dataset Quality and Bias Analysis

### Class Imbalance

The dataset contains different numbers of images for each class.

Fresh and Grenning have more examples than Blackspot and Canker. This imbalance can affect the ability of a model to recognize less-represented categories.

### Visual Similarity

Blackspot and Canker were occasionally confused by both models. This suggests that their visual characteristics may be similar in some images.

### Image Variation

Image dimensions and file sizes vary. Different image sizes, backgrounds, lighting conditions, and camera settings may affect model performance.

### Potential Data Leakage

Duplicate or near-duplicate images across training and test sets can inflate evaluation results.

The dataset analysis checks exact duplicate files using hashes. Near-duplicate images and duplicate content across different files may require additional visual-similarity checks.

### Recommended Improvements

* Collect more examples from underrepresented classes.
* Verify and remove duplicates across dataset splits.
* Investigate label quality.
* Use stratified training and validation splits where appropriate.
* Test the models on independently collected images.
* Evaluate precision, recall, and F1-score for every class.

---

## 13. Production Performance

The observed results are promising, but the models are not yet proven ready for production.

Before deployment, they should be tested on images captured under different real-world conditions, including:

* Different lighting conditions.
* Different camera devices.
* Different backgrounds.
* Different fruit orientations.
* Different levels of damage or ripeness.
* Images collected from previously unseen sources.

Production evaluation should also measure:

* Inference latency.
* Memory usage.
* Per-class precision and recall.
* Failure rates.
* Performance on out-of-distribution images.

The reported test-set accuracy should not be treated as a guarantee of production accuracy.

---

## 14. Drawbacks and Limitations

1. **Limited dataset:** The available dataset contains 1,090 images and may not represent all real-world conditions.
2. **Class imbalance:** The number of images differs across categories.
3. **Class confusion:** Blackspot and Canker were occasionally misclassified.
4. **Lighting sensitivity:** HSV-based features can be affected by illumination and camera settings.
5. **Validation issue:** The current MobileNetV2 script uses test data for validation during training.
6. **Generalization:** Performance on the current dataset may not transfer to new environments.
7. **Computational requirements:** MobileNetV2 requires more computation than the handcrafted-feature approach.
8. **Dataset mismatch:** The provided implementation uses orange condition classes, while the assignment description refers to avocado ripening stages.

---

## 15. Improvements With More Time

The following improvements could increase reliability and make the project more suitable for deployment:

* Confirm the correct avocado dataset with Greenpod Labs.
* Collect more images from diverse real-world environments.
* Balance underrepresented classes.
* Create separate training, validation, and test sets.
* Keep the final test set completely independent during model development.
* Apply image augmentation.
* Fine-tune selected MobileNetV2 layers.
* Perform systematic hyperparameter tuning.
* Compare per-class precision, recall, and F1-score.
* Test on independently collected real-world images.
* Build a simple web interface for image uploads and predictions.
* Measure inference time and optimize deployment.
* Save model versions and document reproducible experiments.

---

## 16. Where Most of the Time Was Spent

Most of the development time was spent on:

* Understanding the dataset structure.
* Resolving file paths and dataset-loading issues.
* Analyzing image distributions and image properties.
* Implementing image preprocessing and feature extraction.
* Training and evaluating both models.
* Debugging prediction commands.
* Comparing confusion matrices and classification results.

---

## 17. Conclusion

This project implements two approaches for fruit-image classification: Random Forest using HSV color features and MobileNetV2 using transfer learning.

Random Forest achieved 96.97% observed accuracy, while MobileNetV2 achieved 95.96% on the available 99-image test set.

These results are preliminary and should not be interpreted as proof of production readiness. A separate validation set and an untouched test set are needed for a more reliable comparison.

The dataset mismatch must also be resolved before the implementation can be considered a complete solution to the avocado ripening-stage assignment.

---

## 18. Project Files

| File                  | Description                                        |
| --------------------- | -------------------------------------------------- |
| `dataset_analysis.py` | Analyzes dataset distribution and image properties |
| `train_model1.py`     | Extracts HSV features and trains Random Forest     |
| `train_model2.py`     | Trains the MobileNetV2 model                       |
| `predict.py`          | Predicts the class of an input image               |
| `requirements.txt`    | Lists the project's Python dependencies            |
| `outputs/`            | Stores generated analysis and output artifacts     |

---

## Author

# NeerugattiKarthik

**Project:** Computer Vision Classification

**Assignment:** Greenpod Labs Data Science Assignment 2
