

---

# Deep Face Recognition using FaceNet, Keras, Dlib and OpenCV

### Author: **Jitendra Kumar**

**Data Scientist**
M.Tech – IIT Kanpur | B.Tech – NIT Surat

---

## 📌 Project Overview

Face recognition is a computer vision technique that identifies a person from images or video frames.
A typical face recognition system works by:

* Detecting faces in an image
* Extracting meaningful features from each face
* Comparing those features with a database of known faces
* Assigning an identity based on similarity

If the similarity score is below a certain threshold, the face is labeled as **unknown**.
The task of comparing two faces to determine whether they belong to the same person is known as **face verification**.

---

## 🧠 Approach Used

This project implements a deep learning–based face recognition pipeline using:

* **FaceNet-based CNN model** for feature extraction
* **Keras** for neural network implementation
* **Dlib** for face detection and landmark estimation
* **OpenCV** for image transformation and alignment

The methodology follows the FaceNet research approach with modifications inspired by the OpenFace project.

---

## 🚀 Key Features

This project demonstrates how to:

1. **Detect and align faces** from input images
2. **Generate 128-dimensional embeddings** for each face
3. **Compare embeddings using Euclidean distance**
4. Perform face recognition using:

   * K-Nearest Neighbors (KNN)
   * Support Vector Machine (SVM)
5. Visualize embedding clusters using t-SNE

---

# 🧩 CNN Architecture

The neural network used in this project is a variant of the **Inception architecture**, specifically:

* The **nn4.small2** model from OpenFace
* Implemented in Keras
* Outputs a **128-dimensional embedding vector**

### Important Points

* Final layers include:

  * A fully connected layer with **128 units**
  * Followed by **L2 normalization**
* These form the **embedding layer**
* Embeddings represent faces in a compact feature space

Model can be created as:

```python
from model import create_model
model = create_model()
```

---

# 🎯 Training Objective – Triplet Loss

Instead of standard classification loss, FaceNet uses **Triplet Loss**.

### Goal:

* Bring embeddings of the **same person closer**
* Push embeddings of **different people farther apart**

Triplets consist of:

* **Anchor** – reference image
* **Positive** – same identity
* **Negative** – different identity

### Simplified Idea

```
loss = max(distance(anchor, positive) 
           - distance(anchor, negative) 
           + margin, 0)
```

This ensures meaningful and discriminative embeddings.

---

# 🛑 Why We Use a Pre-trained Model

Training such a network from scratch requires:

* Millions of images
* Huge computational resources
* Weeks of GPU training

For example, the original FaceNet model was trained on:

* **200 million images**
* **8 million identities**

### Therefore, we use a Pre-trained Model

* Provided by the OpenFace project
* Converted to Keras-compatible format
* Can be loaded directly:

```python
model.load_weights('weights/nn4.small2.v1.h5')
```

---

# 📂 Custom Dataset

For demonstration, a small dataset is used:

* Subset of the **LFW dataset**
* 100 images
* 10 different identities
* 10 images per person

The dataset can easily be replaced with your own images.

---

# 🔧 Face Alignment

The model expects **aligned face images**.

Alignment is performed using:

* **Dlib** – for face detection and landmarks
* **OpenCV** – for transformation and cropping

All faces are converted to:

* **96 × 96 RGB images**
* Standardized pose and scale

Example alignment function:

```python
def align_image(img):
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img),
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
```

---

# 🧬 Generating Face Embeddings

Once images are aligned, embeddings are generated:

```python
embedded[i] = model.predict(preprocessed_image)
```

Each face is converted into a **128-dimensional vector**.

---

# 📏 Distance Threshold Selection

To decide whether two faces are the same person, we compute:

* Euclidean distance between embeddings
* Compare against a threshold value

### Evaluation Strategy

* Compute distances for all image pairs
* Measure performance using **F1 score** (better than accuracy for imbalanced data)
* Choose threshold with highest F1 score

Example result:

* Optimal threshold ≈ **0.56**
* Verification accuracy ≈ **95–96%**

---

# 👤 Face Recognition

Once embeddings are available, recognition can be done in two ways:

### 1. Direct Distance Matching

* Compare input embedding with database
* Assign identity with smallest distance
* If distance > threshold → **Unknown**

### 2. Classifier-Based Recognition

Use machine learning classifiers on embeddings:

* **KNN Classifier**
* **SVM Classifier**

Example performance on test data:

* **KNN Accuracy: ~96%**
* **SVM Accuracy: ~98%**

---

# 📊 Dataset Visualization

t-SNE is used to project embeddings into 2D space.

* Each identity forms a separate cluster
* Shows how well the model separates faces

```python
from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2).fit_transform(embedded)
```

This confirms that embeddings are highly discriminative.

---

# 🛠 Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* Dlib
* NumPy
* scikit-learn
* Matplotlib

---

# 📁 Project Structure

```
├── images/                  # Dataset images
├── models/                  # Pre-trained models
├── weights/                 # Converted weight files
├── align.py                 # Face alignment module
├── model.py                 # CNN architecture
├── Face_Recognition.ipynb   # Main notebook
├── requirements.txt
└── README.md
```

---

# 📝 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Pre-trained Weights

Place weight files inside:

```
weights/
```

### 3. Run Notebook

```
Face_Recognition.ipynb
```

---

# 🔮 Future Enhancements

* Real-time webcam recognition
* Attendance system
* GUI application
* Database integration
* Mobile deployment

---

# 📜 References

1. **FaceNet: A Unified Embedding for Face Recognition and Clustering**
   [https://arxiv.org/abs/1503.03832](https://arxiv.org/abs/1503.03832)

2. **Going Deeper with Convolutions**
   [https://arxiv.org/abs/1409.4842](https://arxiv.org/abs/1409.4842)

---

## 📌 Acknowledgment

This project is inspired by:

* OpenFace
* Keras-OpenFace
* Dlib and OpenCV libraries

---

### 👨‍💻 Author

**Jitendra Kumar**
Data Scientist
M.Tech – IIT Kanpur
B.Tech – NIT Surat

---

### Happy Coding! 🚀

---
