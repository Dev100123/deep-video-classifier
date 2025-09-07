# 🎥 Deep Video Classification (TensorFlow + CNN3D)

This project implements a **deep learning pipeline for video classification** using **TensorFlow, Conv3D, and an OOP (Object-Oriented) architecture**.  
It loads `.mp4` videos, extracts frames, builds datasets, trains a 3D CNN model, and evaluates classification accuracy.

---

## 🚀 Features
- Modular **OOP-style codebase** (easy to extend & maintain)
- Automatic video frame extraction (`cv2 + TensorFlow`)
- Train/Validation split with `scikit-learn`
- 3D Convolutional Neural Network (`Conv3D`)
- Training history plots (loss & accuracy) saved automatically
- Configurable hyperparameters in one place (`config.py`)

---

## 📂 Project Structure
<code>
video_classification/
│
├── config.py # Configurations (classes, paths, hyperparameters)
├── utils.py # Utility functions (frame extraction, formatting)
├── data_loader.py # Video dataset loader (train/val split, tf.data pipelines)
├── model_builder.py # CNN3D model architecture + pre trained model (VGG16, VGG19, & ResNet50)
├── trainer.py # Training, evaluation & plot saving
├── main.py # Entry point
└── plots/ # Training history plots (auto-generated)
</code>

## 📊 Dataset
Organize your dataset like this:
<code>
dataset/
│
├── class1/
│   ├── video1.mp4
│   ├── video2.mp4
│
├── class2/
│   ├── video3.mp4
│   ├── video4.mp4
</code>
---

## ⚙️ Requirements
Install dependencies:
```python
pip install -r requirements.txt
```
## ▶️ Run Training
```python
python main.py
```
<code>
This will:
  ->Load videos and extract frames
  ->Create train/validation datasets
  ->Train the model (saving the best checkpoint to model.h5)
  ->Save training plots in plots/
  ->Print validation accuracy
</code>


