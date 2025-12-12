# Neural Networks & Deep Learning - Laboratory Exercises

This repository contains a comprehensive collection of solutions and reports for a **Neural Networks** course. The projects span from manual implementations of core machine learning algorithms (from scratch) to advanced deep learning architectures using modern frameworks like **PyTorch** and **TensorFlow/Keras**.

## Tech Stack & Tools

* **Languages:** Python
* **Data Manipulation:** NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn
* **Deep Learning Frameworks:** PyTorch, TensorFlow, Keras
* **Machine Learning:** Scikit-learn

---

## Project Overview

### Lab 1: Exploratory Data Analysis (EDA)
**Goal:** Introduction to data preprocessing using the *Heart Disease* dataset.
* **Key Activities:**
    * Analyzed class balance and feature distributions.
    * Performed Shapiro-Wilk tests for normality.
    * Implemented strategies for handling missing values and categorical data.
* **Outcome:** Prepared a clean, normalized feature matrix ready for model ingestion.

### Lab 2: Logistic Regression (From Scratch)
**Goal:** Understanding the mathematical foundations of a single-layer neural network.
* **Implementation:**
    * Manual implementation of the **Sigmoid** activation function.
    * Derivation of the **Cross-Entropy Loss** function.
    * Implementation of **Gradient Descent** for weight updates.
* **Outcome:** Successfully trained a model to classify heart disease presence, verified using Accuracy, Precision, Recall, and F1-score metrics.

### Lab 3: Multi-Layer Perceptron & Backpropagation (From Scratch)
**Goal:** Deep dive into the mechanics of neural networks without auto-differentiation tools.
* **Implementation:**
    * Built a fully connected Neural Network class.
    * Implemented **Forward** and **Backward Propagation** manually (chain rule application).
    * Implemented ReLU and Sigmoid activation derivatives.
* **Experiments:** Extensive Grid Search on hidden layer sizes, learning rates, weight initialization strategies, and data normalization.
* **Key Findings:** Data normalization and correct weight initialization (e.g., small standard deviation) are critical for model convergence.

### Lab 4: PyTorch Implementation & Optimization
**Goal:** Transitioning from manual code to the **PyTorch** framework.
* **Key Activities:**
    * Recreated the MLP architecture using `torch.nn.Module`.
    * Utilized `DataLoader` for efficient batch processing.
* **Experiments:** Comparison of optimizers (**SGD, Adam, RMSprop**) and batch sizes.
* **Key Findings:** The **Adam** optimizer generally provided the fastest convergence and best stability compared to standard SGD.

### Lab 5: Computer Vision with MLPs (FashionMNIST)
**Goal:** Image classification using fully connected networks.
* **Key Activities:**
    * Preprocessing image data (FashionMNIST).
    * Comparing single-layer vs. two-layer architectures.
* **Robustness Testing:** Analyzed the model's performance when injecting **Gaussian noise** into training vs. testing data.
* **Key Findings:** Models trained on noisy data showed significantly higher robustness when facing noisy test data, highlighting the importance of data augmentation.

### Lab 6: Convolutional Neural Networks (CNN)
**Goal:** Implementing CNNs for improved image classification performance.
* **Architecture:** Utilized `Conv2d` layers, `MaxPool2d`, and `LazyLinear` layers in PyTorch.
* **Experiments:**
    * Tuning kernel sizes and the number of output channels.
    * Analyzing the impact of pooling layer dimensions.
* **Key Findings:** CNNs significantly outperformed MLPs on image data. Heatmaps generated during the analysis showed that specific combinations of kernel sizes and channel depth maximize accuracy.

### Lab 7: Recurrent Neural Networks (NLP & Sentiment Analysis)
**Goal:** Sequence processing using RNNs and LSTMs on the **IMDB** dataset.
* **Architecture:** Implemented `Embedding` layers followed by `SimpleRNN` or `LSTM` units using **TensorFlow/Keras**.
* **Experiments:**
    * **RNN vs. LSTM:** Comparison of long-term dependency handling.
    * **Sequence Length:** Analyzed the trade-off between sequence truncation (information loss) and training speed.
* **Key Findings:** LSTMs drastically outperformed SimpleRNNs. There is a distinct trade-off between the maximum sequence length (padding/truncating) and training time/accuracy.

---

## 🚀 How to Run

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    ```
2.  Install dependencies:
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn torch torchvision tensorflow
    ```
3.  Navigate to the specific lab folder and run the script:
    ```bash
    python lab_04_pytorch_optimization.py
    ```

---

## 📈 Visualizations
The repository includes generated plots (loss curves, accuracy comparisons, heatmaps) in the `plots/` directory or embedded within the exercise notebooks to visualize the training process and experimental results.

---
*Created as part of the Neural Networks course curriculum.*
