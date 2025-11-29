# 🔊 Speaker Diarization – Final Machine Learning Project  
### Custom ResNet50 Embedding Model + Clustering  
### Author: Antonella Ríos  

---

## 🧠 Overview

This project implements **Speaker Diarization**, answering “who spoke when?” using a **fully custom machine learning pipeline**, without PyAnnote end-to-end models.

The system includes:

- Custom **ResNet50** audio embedding network  
- Mel-spectrogram generation  
- Sliding-window segmentation  
- Embedding extraction  
- Agglomerative clustering  
- DER/JER evaluation  
- External audio inference  

Everything is coded manually using **PyTorch, Librosa, NumPy, Scikit-Learn**.

---

## 🎯 Project Goals

- Train a custom ResNet50 to generate speaker embeddings  
- Build diarization from scratch with clustering  
- Compute DER & JER on VoxConverse  
- Support inference on external MP3/WAV  
- Ensure reproducibility with Drive + checkpoints  

---

## 📁 Project Structure

~~~plaintext
TP_FINAL_DIARIZATION/
├── datasets/
│   └── voxconverse/
│       ├── audio/
│       ├── rttm/
│       └── metadata/
│
├── checkpoints/
│   ├── model/
│   ├── embeddings/
│   ├── training_logs/
│   └── clustering/
│
├── notebooks/
│   └── diarization_pipeline.ipynb
│
└── results/
    ├── diarization_outputs/
    ├── rttm_predictions/
    └── metrics/
~~~

---

## 🔧 Technical Pipeline (exact implementation)

### **1. Audio Preprocessing**
- Loaded with librosa  
- Resampled to 16 kHz  
- Converted to mono  
- Sliding windows:
  - Window: **1.5 s**
  - Hop: **0.75 s**

---

## **2. Mel-Spectrogram Generation**
Consistent preprocessing:

- Mel filterbank  
- Log-scaled  
- Normalized  
- Ready for CNN input  

---

## **3. Custom ResNet50 Embedding Model**

The ResNet50 has been **modified**:

- First Conv layer changed to accept **1-channel** input  
- Global feature map → Flatten  
- Fully Connected layer **2048 → 512**  
- BatchNorm(512)  
- Dropout(0.3)  
- L2-normalized embeddings  

Optimized with:

- Adam optimizer  
- CrossEntropy loss  
- Optional AMP (mixed precision)  
- Epoch logging and validation  

Checkpoint saved in:

~~~plaintext
/checkpoints/model/resnet50_speaker_embeddings.pth
~~~

---

## **4. Training Process**
Using VoxConverse speaker segments:

- Balanced batches  
- Spectrogram augmentation  
- Validation on each epoch  
- GPU training (Colab)  

---

## **5. Embedding Extraction**
For each audio file:

- Apply sliding windows  
- Convert window → mel-spec  
- Feed to ResNet50  
- Store embeddings:

~~~plaintext
/checkpoints/embeddings/{audio_id}.npy
~~~

---

## **6. Clustering**
Agglomerative clustering (cosine affinity):

~~~plaintext
/checkpoints/clustering/{audio_id}.npy
~~~

Creates the speaker groups that form diarization.

---

## **7. Diarization Assembly**
- Map window→speaker label  
- Merge continuous segments  
- Export RTTM-style prediction:

~~~plaintext
/results/rttm_predictions/{audio_id}.rttm
~~~

---

## 🧪 Evaluation (DER & JER)

Two metrics implemented:

- **DER (Diarization Error Rate)**
- **JER (Jaccard Error Rate)**

Formula:

~~~plaintext
DER = (False Alarm + Missed Speech + Confusion) / Total Speech
~~~

Results saved in:

~~~plaintext
/results/metrics/{audio_id}_metrics.json
~~~

---

## 🐞 Challenges & Solutions

### ✔ Colab resets  
→ Checkpoints in Drive + `os.path.exists()` guards

### ✔ Long training/inference  
→ Windowed processing + smaller test subsets

### ✔ RTTM alignment issues  
→ Manual inspection + consistent loader logic

### ✔ Model compatibility  
→ Forced `numpy==1.26.4` and correct Torch+Librosa versions  

---

## 🎧 External Audio Inference

Your model supports **any MP3/WAV**:

- Load file  
- Window segmentation  
- Spectrogram  
- Embedding extraction  
- Clustering  
- Final diarization output  

Example console output:

~~~plaintext
Speaker 0 → 00:00:00 - 00:00:12
Speaker 1 → 00:00:12 - 00:00:28
Speaker 0 → 00:00:28 - 00:00:41
~~~

Example RTTM:

~~~plaintext
file1 1 SPEAKER 0.00 12.00 <NA> <NA> 0 <NA> <NA>
file1 1 SPEAKER 12.00 16.00 <NA> <NA> 1 <NA> <NA>
~~~

---

## 🧩 How to Run This Project

### **1. Mount Drive**
~~~python
from google.colab import drive
drive.mount('/content/drive')
~~~

### **2. Install Dependencies**
~~~bash
pip install torch torchvision torchaudio
pip install librosa
pip install scikit-learn
pip install numpy==1.26.4
~~~

### **3. Run the Notebook Sections**
1. Imports & utils  
2. Preprocessing  
3. Spectrograms  
4. ResNet50 model  
5. Training  
6. Embeddings  
7. Clustering  
8. Diarization  
9. Evaluation  
10. External audio test  

---

## 👩‍💻 Author

**Antonella Ríos**  
Junior Data Analyst & Data Scientist Trainee  
📍 Salta, Argentina  
📧 antonella.datasolutions@gmail.com  
🔗 linkedin.com/in/antonellarios  

---


---


Antonella Ríos
Junior Data Analyst & Data Scientist (Trainee)
📍 Salta, Argentina
📧 antonella.datasolutions@gmail.com

🔗 linkedin.com/in/antonellarios
