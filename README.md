# HandSignDetection

A Python project for American Sign Language (ASL) hand sign detection and data collection using
OpenCV, cvzone, and MediaPipe.

## Features

- Capture hand images from your webcam
- Detect and crop hand regions using
  [`HandDetector`](https://github.com/cvzone/cvzone/blob/master/cvzone/HandTrackingModule.py) from
  cvzone
- Automatically resize and center hand images for dataset creation
- Save processed images to the `Data/A` folder for training or analysis

## Requirements

- Python 3.11
- opencv-python
- cvzone
- mediapipe
- numpy
- streamlit

## Setup

First, create and activate a virtual environment:

```sh
uv venv
source venv/bin/activate
```

Then, install the required packages:

```sh
uv sync
```

## Usage

To start capturing hand images, run the following command:

Press `q` to quit the image capture window.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug
fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
