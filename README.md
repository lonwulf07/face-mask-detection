# 😷 Face Mask Detection using Deep Learning

A beginner-friendly Computer Vision project that detects whether a
person is wearing a face mask or not using a Convolutional Neural
Network (CNN) built with TensorFlow/Keras and deployed using Streamlit.

------------------------------------------------------------------------

# 📌 Project Overview

This project classifies images into two categories:

-   With Mask
-   Without Mask

It uses a trained deep learning model and provides an interactive web
interface where users can upload images and get predictions with
confidence scores.

------------------------------------------------------------------------

# 🚀 Features

-   Deep Learning model built using TensorFlow/Keras
-   Image classification for mask detection
-   Streamlit web app for easy interaction
-   Clean and modular project structure
-   Beginner‑friendly and resume‑ready project

------------------------------------------------------------------------

# 🧠 Model Training

The model training was performed using Google Colab because the local
system did not have a dedicated GPU.

The notebook is available in:

    notebooks/face_mask_training.ipynb

Google Colab provided:

-   Free GPU acceleration
-   Faster training
-   Better performance

------------------------------------------------------------------------

# 📁 Project Structure

    face-mask-detection/
    │
    ├── dataset/
    │   ├── with_mask/
    │   └── without_mask/
    │
    ├── models/
    │   └── face_mask_model.h5
    │
    ├── notebooks/
    │   └── face_mask_training.ipynb
    │
    ├── src/
    │   └── predict.py
    │
    ├── app/
    │   └── app.py
    │
    ├── requirements.txt
    ├── README.md

------------------------------------------------------------------------

# ⚙️ Installation

## Step 1: Clone Repository

    git clone https://github.com/lonwulf07/face-mask-detection.git
    cd face-mask-detection

------------------------------------------------------------------------

## Step 2: Create Virtual Environment

Python version used:

Python 3.10.11

Create environment:

    python -m venv venv

Activate:

Windows:

    venv\Scripts\activate

------------------------------------------------------------------------

## Step 3: Install Requirements

    pip install -r requirements.txt

------------------------------------------------------------------------

# ▶️ Run Streamlit App

    streamlit run app/app.py

------------------------------------------------------------------------

# 🖼️ App Preview

Upload image → Get Prediction → Confidence Score

Example:

Prediction: With Mask\
Confidence: 97.45%

------------------------------------------------------------------------

# 🧪 Technologies Used

-   Python
-   TensorFlow / Keras
-   OpenCV
-   NumPy
-   Streamlit
-   Google Colab

------------------------------------------------------------------------

# 📈 Future Improvements

-   Real‑time detection
-   Face detection integration
-   Deploy to cloud

------------------------------------------------------------------------

# 👨‍💻 Author

Mohit Sharma

------------------------------------------------------------------------

# ⭐ If you like this project, consider giving it a star!
