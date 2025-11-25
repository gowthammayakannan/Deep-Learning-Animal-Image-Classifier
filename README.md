# 🐾 AnimalVision-CNN

AnimalVision-CNN is a deep learning project built to classify animal images using a custom Convolutional Neural Network. It includes dataset extraction, preprocessing, augmentation, model training, evaluation, visual metrics, and prediction on new images.

## 🚀 Features
- Automatic ZIP dataset extraction
- Train/validation split using ImageDataGenerator
- CNN architecture with Conv2D, MaxPooling2D, Dropout
- Accuracy & loss visualization
- Prediction on random test images
- Easily customizable for any image dataset

## 📁 Project Structure
AnimalVision-CNN/
│── data/
│   └── Animals.zip  
│── notebooks/
│   └── training_notebook.ipynb
│── src/
│   ├── train.py
│   ├── predict.py
│   └── utils.py
│── models/
│   └── animal_model.h5
│── README.md
│── requirements.txt

## 🧠 Technologies Used
- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib
- ImageDataGenerator

## 🔧 How to Run
1. Place your dataset in `data/Animals.zip`
2. Install dependencies:
