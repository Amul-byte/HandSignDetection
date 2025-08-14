import cv2
import streamlit as st
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import os

st.set_page_config(page_title="Hand Sign Detection", layout="wide")
st.title("🤟 Hand Sign Detection App")

# Load model and labels
classifier = Classifier("model/keras_model.h5", "model/labels.txt")
detector = HandDetector(maxHands=1)
labels = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "X", "Y", "Z", "W"
]

imgSize = 400
offset = 30
folder = "Data/Z"
os.makedirs(folder, exist_ok=True)
counter = 0

run = st.checkbox('Start Webcam')
save_img = st.checkbox('Save Captured Image')

frame_placeholder = st.empty()
crop_placeholder = st.empty()
white_placeholder = st.empty()

if run:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            st.warning("Could not access camera.")
            break

        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            aspect_ratio = h / w
            if aspect_ratio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wGap + wCal] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hGap + hCal, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite)
            label = labels[index]
            cv2.putText(imgOutput, label, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 2)

            if save_img:
                counter += 1
                filename = f"{folder}/Image_{int(time.time())}.jpg"
                cv2.imwrite(filename, imgWhite)
                st.success(f"Saved {filename}")

            # Display the image crops
            crop_placeholder.image(cv2.cvtColor(imgCrop, cv2.COLOR_BGR2RGB), caption="Cropped Image", use_container_width =True)
            white_placeholder.image(cv2.cvtColor(imgWhite, cv2.COLOR_BGR2RGB), caption="Resized to 400x400", use_container_width =True)

        frame_placeholder.image(cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB), caption="Live Webcam Feed", channels="RGB")

    cap.release()
else:
    st.info("✅ Click the checkbox above to start the webcam.")
