# Use Python base image
FROM python:3.11-slim
# This line specifies the base image for your Docker container.
# `python:3.11-slim` is a good choice as it's a lightweight Python image,
# which helps keep your final Docker image size smaller compared to full Python images.

# Set working directory
WORKDIR /app
# This sets the working directory inside the container to `/app`.
# All subsequent commands (like COPY, RUN, CMD) will be executed relative to this directory.

# Install system dependencies (libGL for OpenCV, and additional libs for MediaPipe)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libxcb-randr0 \
    libxcb-xfixes0 \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*
# This section is crucial for resolving shared library errors.
# `apt-get update` refreshes the list of available packages.
# `apt-get install -y` installs the specified packages without prompting for confirmation.
# `libgl1` provides the `libGL.so.1` library (for OpenCV).
# `libglib2.0-0` is a common dependency for graphical libraries.
# `libxcb-randr0` and `libxcb-xfixes0` are often required by MediaPipe for its underlying dependencies.
# `libffi-dev` is sometimes needed for certain Python packages that rely on C extensions.
# `&& rm -rf /var/lib/apt/lists/*` cleans up the apt cache, reducing the image size.

# Upgrade pip and install Python dependencies
COPY requirements.txt .
# This copies your `requirements.txt` file from your local project directory
# into the `/app` directory inside the container. It's copied early to
# leverage Docker's build cache: if requirements.txt doesn't change,
# this layer (and subsequent ones) won't be rebuilt.
RUN pip install --upgrade pip
# Upgrades pip to its latest version, which is good practice.
RUN pip install -r requirements.txt
# Installs all Python packages listed in your `requirements.txt`.
# Ensure that if you're using `opencv-python`, you consider replacing it
# with `opencv-python-headless` in your `requirements.txt` for server deployments,
# as it's designed for environments without a display and is generally smaller.
# Also, ensure `mediapipe` is listed in your `requirements.txt`.

# Copy all project files
COPY . .
# This copies the rest of your project files from your local directory
# into the `/app` directory inside the container.

# Expose Streamlit port
EXPOSE 8501
# This informs Docker that the container will listen on port 8501 at runtime.
# It's primarily for documentation and network configuration (e.g., for port mapping).

# Command to run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
# This defines the default command that will be executed when the container starts.
# It runs your Streamlit app, binding it to port 8501 and making it accessible
# from any network interface (0.0.0.0) within the container.
