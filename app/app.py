import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# --------------------------
# Page Config
# --------------------------

st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="centered"
)

st.title("😷 Face Mask Detection App")

st.write("Upload an image to detect whether the person is wearing a mask or not.")

# --------------------------
# Load Model
# --------------------------

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("../models/mask_detector.h5")
    return model

model = load_model()

# --------------------------
# Upload Image
# --------------------------

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------
# Prediction Function
# --------------------------

def predict(image):

    image = image.convert("RGB")

    image = image.resize((224, 224))

    image = np.array(image)

    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)

    return prediction


# --------------------------
# Show Result
# --------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", width="stretch")

    st.write("")

    st.write("Predicting...")

    prediction = predict(image)

    confidence = float(prediction[0][0])

    if confidence > 0.5:

        result = "Without Mask"

    else:

        result = "With Mask"
        confidence = 1 - confidence

    st.success(f"Prediction: {result}")

    st.info(f"Confidence: {confidence*100:.2f}%")
