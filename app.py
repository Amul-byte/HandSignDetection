# app.py
import os, io, math, time
import numpy as np
import cv2
import streamlit as st

# Reduce noisy logs (place before importing TF if you use it elsewhere)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

# --- Third-party for webcam-in-browser ---
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- Your existing modules ---
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from PIL import Image

st.set_page_config(page_title="🤟 Hand Sign Detection", layout="wide")
st.title("🤟 Hand Sign Detection App")

# =========================
# Models & constants
# =========================
@st.cache_resource
def load_detector_and_classifier():
    # Load once per process (cached across Streamlit reruns)
    clf = Classifier("model/keras_model.h5", "model/labels.txt")
    det = HandDetector(maxHands=1)
    return det, clf

detector, classifier = load_detector_and_classifier()

LABELS = [
    "A","B","C","D","E","F","G","H","I","J",
    "K","L","M","N","O","P","Q","R","S","T",
    "U","V","X","Y","Z","W"
]

# =========================
# Sidebar controls
# =========================
with st.sidebar:
    st.header("Settings")
    OFFSET = st.slider("Crop offset (px)", 0, 60, 30, 2)
    IMG_SIZE = st.select_slider("Classifier canvas size", [224, 300, 400], value=400)
    PROCESS_EVERY = st.slider("Process every Nth frame (lower = heavier CPU)", 1, 5, 1, 1)
    show_debug = st.checkbox("Show debug panels (crop & normalized)")
    st.divider()
    st.caption("Snapshots are saved to the app filesystem (ephemeral on Streamlit Cloud).")
    enable_snapshots = st.checkbox("Enable snapshots")
    SNAP_DIR = "Data/Z"
    os.makedirs(SNAP_DIR, exist_ok=True)

mode = st.radio(
    "Mode",
    ["Live (browser camera — recommended)", "Single photo", "Local webcam (dev only)"],
    index=0,
    help="On Streamlit Cloud, use 'Live' or 'Single photo'. Local webcam works only on your machine."
)

# =========================
# Core processing
# =========================
def normalize_and_classify(bgr_img, bbox, offset, img_size):
    """
    Given a BGR image and a bbox (x,y,w,h), clamp ROI, place onto a white square canvas,
    classify with your cvzone Classifier, and return:
      - img_output (annotated), img_crop, img_white, predicted_label (or None)
    """
    img = bgr_img
    img_output = img.copy()

    x, y, w, h = bbox
    H, W = img.shape[:2]

    # Clamp bbox with offset to image bounds
    x1 = max(0, x - offset)
    y1 = max(0, y - offset)
    x2 = min(W, x + w + offset)
    y2 = min(H, y + h + offset)

    img_crop = img[y1:y2, x1:x2]
    if img_crop.size == 0:
        return img_output, None, None, None

    # White canvas normalization (preserve aspect)
    img_white = np.ones((img_size, img_size, 3), np.uint8) * 255
    hC, wC = img_crop.shape[:2]
    aspect = hC / float(wC)

    if aspect > 1:
        k = img_size / float(hC)
        wCal = max(1, math.ceil(k * wC))
        img_resize = cv2.resize(img_crop, (wCal, img_size))
        wGap = (img_size - wCal) // 2
        img_white[:, wGap:wGap + wCal] = img_resize
    else:
        k = img_size / float(wC)
        hCal = max(1, math.ceil(k * hC))
        img_resize = cv2.resize(img_crop, (img_size, hCal))
        hGap = (img_size - hCal) // 2
        img_white[hGap:hGap + hCal, :] = img_resize

    # Classify
    prediction, index = classifier.getPrediction(img_white)
    label = LABELS[index]

    # Draw UI on output frame
    cv2.putText(img_output, label, (x, max(30, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    cv2.rectangle(img_output, (x1, y1), (x2, y2), (0, 200, 0), 2)

    return img_output, img_crop, img_white, label

def process_frame(bgr_img, offset, img_size):
    """
    Detect hand, classify, and return annotated frame plus intermediates.
    """
    img = bgr_img
    hands, _ = detector.findHands(img)  # we ignore cvzone's annotated image

    if hands:
        hand = hands[0]
        bbox = hand['bbox']  # (x, y, w, h)
        return normalize_and_classify(img, bbox, offset, img_size)
    else:
        return img, None, None, None

# =========================
# LIVE: Browser camera (WebRTC)
# =========================
if mode == "Live (browser camera — recommended)":
    col1, col2 = st.columns(2, vertical_alignment="center")
    snap_btn = col1.button("📸 Save normalized image", disabled=not enable_snapshots)
    status = col2.empty()

    class HandSignTransformer(VideoTransformerBase):
        def __init__(self):
            self.last_imgWhite = None
            self.last_imgCrop = None
            self.last_label = None
            self.last_bgr = None
            self.i = 0
            # Defaults (will be updated from sidebar via ctx.video_processor after creation)
            self.offset = 30
            self.img_size = 400
            self.process_every = 1

        def transform(self, frame: av.VideoFrame) -> np.ndarray:
            bgr = frame.to_ndarray(format="bgr24")

            # Optionally skip frames to reduce CPU
            self.i = (self.i + 1) % self.process_every
            if self.i != 0:
                return bgr

            annotated, imgCrop, imgWhite, label = process_frame(
                bgr, self.offset, self.img_size
            )

            # Keep latest frames for snapshot/debug
            self.last_bgr = annotated
            self.last_imgCrop = imgCrop
            self.last_imgWhite = imgWhite
            self.last_label = label

            return annotated

    rtc_configuration = {
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }

    ctx = webrtc_streamer(
        key="handstream",
        video_processor_factory=HandSignTransformer,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration=rtc_configuration,
    )

    # Sync sidebar settings into the running video processor
    if ctx and ctx.video_processor:
        ctx.video_processor.offset = OFFSET
        ctx.video_processor.img_size = IMG_SIZE
        ctx.video_processor.process_every = PROCESS_EVERY

    # Save snapshot (normalized canvas)
    if snap_btn and ctx and ctx.video_processor and ctx.video_processor.last_imgWhite is not None:
        fname = os.path.join(SNAP_DIR, f"Image_{int(time.time())}.jpg")
        cv2.imwrite(fname, ctx.video_processor.last_imgWhite)
        status.success(f"✅ Saved {fname}")

        # Also offer a download (since Streamlit Cloud storage is ephemeral)
        rgb = cv2.cvtColor(ctx.video_processor.last_imgWhite, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        st.download_button("⬇️ Download snapshot", data=buf.getvalue(),
                           file_name=os.path.basename(fname).replace(".jpg", ".png"),
                           mime="image/png")

    # Optional debug previews
    if show_debug and ctx and ctx.video_processor:
        dbg1, dbg2 = st.columns(2)
        if ctx.video_processor.last_imgCrop is not None:
            dbg1.image(
                cv2.cvtColor(ctx.video_processor.last_imgCrop, cv2.COLOR_BGR2RGB),
                caption="Cropped ROI", use_container_width=True
            )
        if ctx.video_processor.last_imgWhite is not None:
            dbg2.image(
                cv2.cvtColor(ctx.video_processor.last_imgWhite, cv2.COLOR_BGR2RGB),
                caption=f"Normalized {IMG_SIZE}×{IMG_SIZE}", use_container_width=True
            )

# =========================
# SINGLE PHOTO mode
# =========================
elif mode == "Single photo":
    img_file = st.camera_input("Take a picture")
    if img_file:
        img = Image.open(img_file).convert("RGB")
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        annotated, crop, white, label = process_frame(bgr, OFFSET, IMG_SIZE)
        c1, c2, c3 = st.columns(3)
        c1.image(img, caption="Original", use_container_width=True)
        if crop is not None:
            c2.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption="Cropped", use_container_width=True)
        if white is not None:
            c3.image(cv2.cvtColor(white, cv2.COLOR_BGR2RGB), caption=f"Normalized {IMG_SIZE}×{IMG_SIZE}", use_container_width=True)
        if label:
            st.subheader(f"Prediction: **{label}**")

        if enable_snapshots and white is not None and st.button("💾 Save normalized image"):
            fname = os.path.join(SNAP_DIR, f"Image_{int(time.time())}.jpg")
            cv2.imwrite(fname, white)
            st.success(f"Saved {fname}")
            # Also offer download
            rgb = cv2.cvtColor(white, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            st.download_button("⬇️ Download snapshot", data=buf.getvalue(),
                               file_name=os.path.basename(fname).replace(".jpg", ".png"),
                               mime="image/png")
    else:
        st.info("Use the camera above to capture an image.")

# =========================
# LOCAL WEBCAM (dev only)
# =========================
else:  # "Local webcam (dev only)"
    st.warning("This mode works only on your machine (no /dev/video0 on Streamlit Cloud).")
    start = st.checkbox("Start local webcam")
    if start:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not access local webcam.")
        frame_area = st.empty()
        crop_area, white_area = st.columns(2)

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    st.error("Failed to read from webcam.")
                    break
                annotated, crop, white, _ = process_frame(frame, OFFSET, IMG_SIZE)
                frame_area.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                 caption="Local Webcam", channels="RGB", use_container_width=True)
                if show_debug:
                    if crop is not None:
                        crop_area.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                                        caption="Cropped", use_container_width=True)
                    if white is not None:
                        white_area.image(cv2.cvtColor(white, cv2.COLOR_BGR2RGB),
                                         caption=f"Normalized {IMG_SIZE}×{IMG_SIZE}", use_container_width=True)
                # Allow Streamlit to handle UI events
                if not st.session_state.get("_run_local_loop", True):
                    break
                # Gentle frame rate
                time.sleep(0.02)
        finally:
            cap.release()
