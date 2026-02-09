# Deep Face Recognition using FaceNet, Keras, Dlib and OpenCV

### Author: **Er. Jitendra Kumar**

**Data Scientist**
M.Tech – IIT Kanpur | B.Tech – NIT Surat

---

## 📌 Repository Link

🔗 **Project Repository:**
[https://github.com/jpb2022/Face-Recognition-FaceNet-OpenCV-Dlib/blob/main/face-recognition.ipynb](https://github.com/jpb2022/Face-Recognition-FaceNet-OpenCV-Dlib/blob/main/face-recognition.ipynb)

---

## 📌 Project Overview

Face recognition is a powerful computer vision application that identifies or verifies a person from images or video frames.

A typical face recognition system works through the following steps:

* Detecting faces in an image
* Extracting meaningful features from each detected face
* Comparing extracted features with stored face representations
* Assigning an identity based on similarity

If the similarity score is below a predefined threshold, the face is labeled as **unknown**.
The process of determining whether two face images belong to the same person is known as **face verification**.

This project implements a complete end-to-end face recognition pipeline using deep learning.

---

## 🧠 Approach Used

This project follows a modern deep learning–based approach for face recognition using:

* **FaceNet-based CNN architecture** for feature extraction
* **Keras / TensorFlow** for deep learning implementation
* **Dlib** for face detection and landmark localization
* **OpenCV** for image preprocessing and alignment

The overall methodology is inspired by the original FaceNet research with modifications from the OpenFace project.

---

## 🚀 Key Features

This project demonstrates how to:

1. Detect faces from images
2. Align and normalize faces
3. Generate 128-dimensional face embeddings
4. Compare faces using Euclidean distance
5. Perform face recognition using:

   * K-Nearest Neighbors (KNN)
   * Support Vector Machine (SVM)
6. Visualize embedding clusters using t-SNE

---

# 🧩 CNN Architecture

The neural network used in this project is based on:

* A variant of the **Inception architecture**
* Specifically the **nn4.small2** model from OpenFace
* Implemented using Keras
* Produces a **128-dimensional face embedding vector**

### Architectural Highlights

* Final layers include:

  * A fully connected layer with **128 units**
  * An **L2 normalization layer**
* These layers together form the **embedding layer**
* Each face is mapped into a compact numerical representation

Model creation:

```python
from model import create_model
model = create_model()
```

---

# 🎯 Training Objective – Triplet Loss

FaceNet does not use traditional classification loss.
Instead, it uses **Triplet Loss** to learn discriminative embeddings.

### Objective:

* Bring embeddings of the **same person closer**
* Push embeddings of **different people farther apart**

Each training sample consists of:

* **Anchor image** – reference
* **Positive image** – same identity
* **Negative image** – different identity

### Concept:

```
loss = max(distance(anchor, positive) 
           - distance(anchor, negative) 
           + margin, 0)
```

This ensures the model learns a meaningful face representation space.

---

# 🛑 Why a Pre-trained Model is Used

Training a FaceNet model from scratch requires:

* Massive labeled datasets
* Very high computational power
* Long training time

For example, the original FaceNet was trained on:

* **200 million images**
* **8 million unique identities**

### Solution

Instead of training from scratch, we use a **pre-trained model**:

* Provided by OpenFace
* Converted to Keras-compatible format
* Easily loadable:

```python
model.load_weights('weights/nn4.small2.v1.h5')
```

This makes the system practical and efficient.

---

# 📂 Custom Dataset

For demonstration purposes, this project uses:

* A subset of the **LFW (Labeled Faces in the Wild)** dataset
* 100 face images
* 10 different identities
* 10 images per person

You can easily replace this dataset with your own images to build a personalized face recognition system.

---

# 🔧 Face Alignment

The FaceNet model requires **aligned faces** for best performance.

Alignment is performed using:

* **Dlib** – face detection and landmark estimation
* **OpenCV** – geometric transformation and cropping

All faces are converted to:

* **96 × 96 RGB format**
* Standardized pose and orientation

Example alignment function:

```python
def align_image(img):
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img),
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
```

---

# 🧬 Generating Face Embeddings

After alignment, embeddings are generated using the CNN:

```python
embedded[i] = model.predict(preprocessed_image)
```

Each face is transformed into a **128-dimensional numerical vector**.

These embeddings are the core of the recognition system.

---

# 📏 Distance Threshold Selection

To decide whether two faces match:

* Compute Euclidean distance between embeddings
* Compare with a learned threshold

### Evaluation Strategy

* Calculate distances for all image pairs
* Evaluate using **F1 score** (better for imbalanced data)
* Select threshold with maximum F1 score

Typical result:

* Optimal threshold ≈ **0.56**
* Verification accuracy ≈ **95–96%**

---

# 👤 Face Recognition Methods

Once embeddings are available, recognition can be performed in two ways:

### 1. Direct Distance Matching

* Compare input embedding with stored embeddings
* Assign identity with smallest distance
* If distance > threshold → **Unknown**

### 2. Classifier-Based Recognition

Use machine learning classifiers:

* **KNN Classifier**
* **SVM Classifier**

Example performance:

* **KNN Accuracy: ~96%**
* **SVM Accuracy: ~98%**

---

# 📊 Dataset Visualization

To understand embedding quality, we use **t-SNE visualization**:

```python
from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2).fit_transform(embedded)
```

This shows:

* Clear identity clusters
* Good separation between different people
* Effectiveness of learned embeddings

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

# 📝 How to Run the Project

### Step 1 – Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2 – Download Pre-trained Weights

Place the weight files inside:

```
weights/
```

### Step 3 – Run the Notebook

Open and execute:

```
Face_Recognition.ipynb
```

---

# 🔮 Future Enhancements

Planned improvements include:

* Real-time webcam face recognition
* Smart attendance system
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

* OpenFace Project
* Keras-OpenFace
* Dlib and OpenCV Libraries

---

### 👨‍💻 Author

**Er. Jitendra Kumar**

**Data Scientist,
M.Tech – IIT Kanpur,
B.Tech – NIT Surat**

---

### Happy Coding! 🚀
