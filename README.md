# 🧠 MNIST Digit Recognition — Project Documentation

A comprehensive study and implementation of Convolutional Neural Networks (CNNs) for handwritten digit recognition using the **MNIST dataset**. This repository includes model training pipelines, comparative experiments, performance analysis, and multiple saved model versions to demonstrate iterative architectural improvements.

---

## 📌 Abstract

Handwritten digit recognition is a fundamental problem in computer vision and pattern recognition. The MNIST dataset serves as a benchmark for evaluating image classification models due to its simplicity and widespread adoption. In this project, we investigate multiple CNN architectures, evaluate their performance, compare training behaviours, and identify the most effective model design. Our best-performing model achieves **up to 99% accuracy**, demonstrating strong generalisation across the dataset.

---

## 📘 1. Introduction

The MNIST dataset consists of **70,000 grayscale images** representing digits from **0–9**, each sized **28×28 pixels**. It has been widely used as a standard dataset for training and testing machine learning models, particularly neural networks.

This project explores:

* Baseline CNN model performance
* Architectural enhancements and their effects
* Hyperparameter tuning
* Comparative analysis of different model versions
* Practical considerations in deploying trained models

---

## 📊 2. Dataset Overview

**MNIST Dataset Characteristics:**

* 60,000 training images
* 10,000 testing images
* Grayscale, 28×28 resolution
* Single-channel pixel intensities (0–255)

### Preprocessing Steps

* Normalization (pixel/255)
* Reshaping to (28, 28, 1)
* One-hot encoding of labels

These steps ensure compatibility with CNN architectures and improve convergence during training.

---

## 🧠 3. Model Architectures

We trained and evaluated **three different CNN architectures**, stored in:

* `mnist_cnn.h5`
* `mnist_cnn.keras`
* `mnist_cnn_v2.keras`

### 3.1 Baseline CNN (mnist_cnn.h5)

* Conv2D (32 filters) + ReLU
* MaxPooling2D
* Flatten
* Dense (128 neurons)
* Softmax output

Performance: **~98% accuracy**

### 3.2 Improved CNN (mnist_cnn.keras)

* Increased filter count
* Better kernel initialization
* Optimised learning rate

Performance: **~98.5% accuracy**

### 3.3 Advanced CNN v2 (mnist_cnn_v2.keras)

* Deeper architecture
* Larger filter sizes
* Enhanced regularisation (dropout / tuning)
* Adam optimizer fine-tuning

Performance: **~99% accuracy**

### Summary of Improvements

| Model           | Accuracy | Training Time | Strengths                | Weaknesses        |
| --------------- | -------- | ------------- | ------------------------ | ----------------- |
| Baseline CNN    | ~98%     | Fast          | Lightweight              | Limited depth     |
| Improved CNN    | ~98.5%   | Moderate      | Better representation    | Slightly slower   |
| Advanced CNN v2 | **99%**  | Highest       | Strongest generalization | Larger model size |

The **Advanced CNN v2** consistently outperformed the others due to deeper feature extraction and optimized hyperparameters.

---

## 🔬 4. Experimental Setup

### 4.1 Training Configuration

* Loss Function: Categorical Cross-Entropy
* Optimizer: Adam
* Batch Size: 32/64
* Epochs: 5–10

### 4.2 Evaluation Metrics

* Accuracy
* Loss curves
* Confusion matrix
* Misclassification analysis

### 4.3 Findings

* Deeper CNNs improve feature abstraction
* Proper normalization accelerates convergence
* Adam outperforms SGD for MNIST-scale datasets
* Overfitting is minimal due to dataset simplicity

---

## 🧪 5. Model Comparison (Research Analysis)

The notebook `model_comparison_mnist.ipynb` includes:

* Side-by-side accuracy curves
* Loss comparison
* Confusion matrix evaluation
* Observations on hyperparameter impact

### Key Insights

* Increasing convolutional depth improves performance more than increasing dense layers
* MaxPooling significantly reduces computational cost while preserving features
* The best model (`mnist_cnn_v2.keras`) demonstrates improved classification stability

---

## 🗂️ 6. Repository Structure

```
mnist-digit-recognition/
│
├── MNIST Project.ipynb              # Full training and visualization
├── mnist_cnn_training.ipynb         # Training pipeline
├── model_comparison_mnist.ipynb     # Comparative study of models
│
├── mnist_cnn.h5                     # Baseline CNN model
├── mnist_cnn.keras                  # Improved model
├── mnist_cnn_v2.keras               # Best performing model
│
├── LICENSE                          # MIT license
└── README.md                        # Documentation
```

---

## ⚙️ 7. How to Run the Project

### Step 1: Clone Repository

```
git clone https://github.com/YOUR_USERNAME/mnist-digit-recognition.git
cd mnist-digit-recognition
```

### Step 2: Install Dependencies

```
pip install -r requirements.txt
```

### Step 3: Run Jupyter Notebooks

```
jupyter notebook
```

Open and execute:

* `MNIST Project.ipynb`
* `mnist_cnn_training.ipynb`
* `model_comparison_mnist.ipynb`

---

## 🔧 8. Using the Saved Models

### Load Model

```python
import tensorflow as tf
model = tf.keras.models.load_model("mnist_cnn_v2.keras")
```

### Predict

```python
import numpy as np
img = np.random.rand(1, 28, 28, 1)
pred = model.predict(img)
print("Predicted Digit:", np.argmax(pred))
```

---

## 📝 9. Conclusion

This research-style project demonstrates that:

* CNNs are highly effective for digit classification tasks
* Deeper architectures outperform shallower ones
* Proper optimization & tuning significantly enhance performance
* The MNIST dataset remains an excellent benchmark for CNN experimentation

The **Advanced CNN v2** model provides the best balance of accuracy, stability, and generalization.

---

## 🤝 10. Contributing

We welcome contributions involving:

* Novel architectures
* Hyperparameter tuning
* Visualization enhancements
* Research extensions

To contribute:

```
1. Fork the repository
2. Create a new branch
3. Commit changes
4. Submit a pull request
```

---

## 📄 11. License

This project is released under the **MIT License**, allowing reuse, modification, and distribution.

---

## ⭐ 12. Support

If this project helped you, please consider:

* ⭐ Starring the repository
* 🔁 Sharing it
* 💬 Suggesting improvements

---

Made with ❤️ for research, experimentation, and learning.
