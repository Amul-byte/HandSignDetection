import streamlit as st
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

# App title
st.title("✋ Real-time Hand Sign Detection")


# Initialize components
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
detector = HandDetector(maxHands=1)
labels = ["A", "C"]
imgSize = 400
offset = 30
FRAME_WINDOW = st.image([])

with st.sidebar:
    st.header("Settings")
    st.write("Adjust the settings below to customize the hand sign detection.")
    run = st.checkbox("Start Webcam")
    
# Start video capture
cap = cv2.VideoCapture(0)

while run:
    success, img = cap.read()
    if not success:
        st.error("Could not access webcam.")
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

        cv2.putText(imgOutput, label, (x, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (0, 255, 0), 2)

    FRAME_WINDOW.image(cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB))

if not run:
    cap.release()
