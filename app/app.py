import streamlit as st
import cv2
import numpy as np
import os
import av
import threading
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="centered"
)

st.title("😷 Face Mask Detection System")


# ============================================
# LOAD MODEL
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "mask_detector.keras")

model = load_model(MODEL_PATH, compile=False)

labels = ["Without Mask", "With Mask"]


# ============================================
# PREDICTION FUNCTION
# ============================================

def predict_image(image):

    img = cv2.resize(image, (224, 224))

    img = img / 255.0

    img = np.reshape(img, (1, 224, 224, 3))

    prediction = model.predict(img, verbose=0)

    confidence = float(np.max(prediction))

    label = labels[np.argmax(prediction)]

    return label, confidence


# ============================================
# SIDEBAR
# ============================================

option = st.sidebar.selectbox(

    "Select Detection Mode",

    (

        "Image Upload",

        "Real-Time Webcam"

    )

)


# ============================================
# IMAGE UPLOAD MODE
# ============================================

if option == "Image Upload":

    st.header("Upload Image")

    uploaded_file = st.file_uploader(

        "Choose an image",

        type=["jpg", "jpeg", "png"]

    )

    if uploaded_file is not None:

        file_bytes = np.asarray(

            bytearray(uploaded_file.read()),

            dtype=np.uint8

        )

        image = cv2.imdecode(file_bytes, 1)

        if image is not None:

            label, confidence = predict_image(image)

            color = (0,255,0) if label=="With Mask" else (0,0,255)

            output = image.copy()

        cv2.putText(

            output,

            f"{label}: {confidence*100:.2f}%",

            (20,40),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            color,

            2

        )

        st.image(output, channels="BGR")

        st.success(f"Prediction: {label}")

        st.info(f"Confidence: {confidence*100:.2f}%")


# ============================================
# REALTIME MODE
# ============================================

elif option == "Real-Time Webcam":

    st.header("Real-Time Webcam Detection")
    
    # RTC Configuration (IMPORTANT — prevents crashes)
    RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )


    # Thread lock for safe prediction
    lock = threading.Lock()


    class VideoProcessor(VideoProcessorBase):

        def recv(self, frame):

            img = frame.to_ndarray(format="bgr24")

            # Resize smaller for performance
            resized = cv2.resize(img, (224, 224))

            resized = resized / 255.0
            reshaped = np.reshape(resized, (1, 224, 224, 3))

            with lock:

                prediction = model.predict(reshaped, verbose=0)

            confidence = float(np.max(prediction))

            label = labels[np.argmax(prediction)]

            color = (0,255,0) if label=="With Mask" else (0,0,255)

            cv2.putText(

                img,

                f"{label}: {confidence*100:.2f}%",

                (20,40),

                cv2.FONT_HERSHEY_SIMPLEX,

                1,

                color,

                2

            )

            return av.VideoFrame.from_ndarray(img, format="bgr24")


    webrtc_streamer(

        key="realtime",

        rtc_configuration=RTC_CONFIGURATION,

        video_processor_factory=VideoProcessor,

        async_processing=True,

    )