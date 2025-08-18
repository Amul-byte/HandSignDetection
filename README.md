# HandSignDetection

A Streamlit app for American Sign Language (ASL) hand-sign detection using **OpenCV**,
**cvzone/MediaPipe**, and a Keras **.h5** classifier.  
This README matches the current `app.py` (WebRTC live mode, single-photo mode, and a local-webcam
dev mode).

---

## Features

- **Live detection in the browser** via `streamlit-webrtc` (no server webcam needed)
- **Single-photo** capture using `st.camera_input`
- Hand **detection & cropping** with `cvzone.HandDetector`
- Square **normalization canvas** (224 / 300 / 400) with aspect-ratio preservation
- **On-demand snapshots** of the normalized image (with a **Download** button)
- Optional **debug panels** (Cropped ROI & normalized canvas)
- (Dev only) **Local webcam** path using `cv2.VideoCapture(0)` on your laptop

---

## Requirements

- Python **3.11**
- Libraries used by `app.py`:
  - `streamlit`
  - `streamlit-webrtc`
  - `opencv-python`
  - `cvzone`
  - `mediapipe`
  - `numpy`
  - `Pillow`
  - `av`
  - **TensorFlow** (required if your classifier is a Keras `.h5` model)
- Model files:
  - `model/keras_model.h5`
  - `model/labels.txt` (class order must match training)
- Optional: **uv** (fast Python package manager) or plain `pip`

> On Streamlit Cloud, files saved to `Data/Z` are **ephemeral** and will be cleared on
> hibernation/redeploy. Use the built-in **Download** button for persistence.

---

## Setup

### With uv (recommended)

```bash
uv venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv sync                            # installs from uv.lock / pyproject / requirements.txt
```
