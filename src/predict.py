# ============================================
# Face Mask Detection - Prediction Script
# ============================================

# Import libraries
import cv2
import numpy as np
from tensorflow.keras.models import load_model


# Load trained model
model = load_model("../models/mask_detector.h5")

print("Model loaded successfully")


# Path to test image
image_path = "../dataset/without_mask/without_mask_11.jpg"   # change if needed


# Read image
image = cv2.imread(image_path)


# Resize to model input size
image_resized = cv2.resize(image, (224, 224))


# Normalize image
image_normalized = image_resized / 255.0


# Reshape image for model
image_reshaped = np.reshape(image_normalized, (1, 224, 224, 3))


# Predict
prediction = model.predict(image_reshaped, verbose=0)[0][0]


# Convert to label
if prediction < 0.5:

    label = "Mask"

else:

    label = "No Mask"


# Print result
print("Prediction:", label)


# Show result on image
cv2.putText(

    image,
    label,
    (20, 40),

    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2

)


cv2.imshow("Face Mask Detection", image)

cv2.waitKey(0)

cv2.destroyAllWindows()