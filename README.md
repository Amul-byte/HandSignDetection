# HandSignDetection

A Python project for American Sign Language (ASL) hand sign detection and data collection using OpenCV, cvzone, and MediaPipe.

## Features

- Captures hand images from webcam
- Detects and crops hand region using [`HandDetector`](https://github.com/cvzone/cvzone/blob/master/cvzone/HandTrackingModule.py) from cvzone
- Automatically resizes and centers hand images for dataset creation
- Saves processed images to the `Data/A` folder for training or analysis

## Requirements

- Python 3.12+
- OpenCV
- cvzone
- MediaPipe
- numpy

Install dependencies with:

```sh
pip install -r requirements.txt
```
