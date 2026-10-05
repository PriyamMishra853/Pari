# PARIKSHAK backend: FastAPI + MediaPipe + YOLO (ONNX) + AprilTag, CPU only.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libGL/glib: OpenCV GUI build (pulled in by mediapipe); libgomp: onnxruntime/torch threads
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch first, so ultralytics does not drag in the 2+ GB CUDA build
RUN pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-server.txt .
RUN pip install -r requirements-server.txt \
    # ultralytics installs opencv-python, mediapipe installs opencv-contrib-python; both own the
    # cv2 module. Keep only the contrib build - it contains the AprilTag (aruco) detector.
    && pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python \
    && pip install --no-deps opencv-contrib-python==5.0.0.93 \
    && python -c "import cv2; cv2.aruco.DICT_APRILTAG_36h11; import mediapipe, onnxruntime; print('vision stack ok', cv2.__version__)"

COPY . .

EXPOSE 8766
CMD ["sh", "-c", "python -m backend.server --host 0.0.0.0 --port ${PORT:-8766} --no-browser"]
