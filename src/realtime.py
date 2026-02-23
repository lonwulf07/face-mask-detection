import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("models/mask_detector.h5")

# Labels
labels = ["With Mask", "Without Mask"]

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Resize for model input
    img = cv2.resize(frame, (224, 224))

    img = img / 255.0
    img = np.reshape(img, (1, 224, 224, 3))

    # Prediction
    prediction = model.predict(img, verbose=0)

    confidence = np.max(prediction)
    label_index = np.argmax(prediction)

    label = labels[label_index]

    text = f"{label}: {confidence*100:.2f}%"

    # Color
    if label == "With Mask":
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    # Show text
    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 2)

    cv2.imshow("Face Mask Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()