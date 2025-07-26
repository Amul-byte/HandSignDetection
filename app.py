import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

st.set_page_config(page_title="Hand Sign Detection")
st.title("✋ Real-time Hand Sign Detection (WebRTC)")

# Load model and set parameters
@st.cache_resource
def load_model():
    return Classifier("model/keras_model.h5", "model/labels.txt")

classifier = load_model()
detector = HandDetector(maxHands=1)
labels = ["A", "C"]
imgSize = 400
offset = 30

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    st.write("Webcam is enabled in your browser.")
    enable = st.checkbox("Start Detection", value=True)

# Video processor (new API)
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        imgOutput = img.copy()
        hands, _ = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            aspect_ratio = h / w

            try:
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

            except Exception as e:
                print("Error:", e)

        return av.VideoFrame.from_ndarray(imgOutput, format="bgr24")

# Start the webcam stream
if enable:
    ctx = webrtc_streamer(
        key="sign-detection",
        video_processor_factory=VideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if not ctx.state.playing:
        st.warning("Click 'Start' above to activate your camera.")
