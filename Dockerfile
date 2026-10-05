# PARIKSHAK backend: FastAPI + MediaPipe + YOLO (ONNX Runtime) + AprilTag. CPU only, no PyTorch.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libGL/glib: OpenCV GUI build; libEGL/GLES: MediaPipe tasks on headless Linux;
# libgomp: onnxruntime threads; libportaudio: imported by mediapipe's audio deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libegl1 libgles2 libgomp1 libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-server.txt .
RUN pip install -r requirements-server.txt \
    # only one package may own the cv2 module - keep the contrib build (it has the AprilTag detector)
    && pip uninstall -y opencv-python opencv-python-headless \
    && pip install --no-deps --force-reinstall opencv-contrib-python==5.0.0.93 \
    && python -c "import cv2; cv2.aruco.DICT_APRILTAG_36h11; import mediapipe, onnxruntime; print('vision stack ok', cv2.__version__)"

COPY . .

# Railway injects PORT; 8766 is the local default.
EXPOSE 8766
CMD ["sh", "-c", "python -m backend.server --host 0.0.0.0 --port ${PORT:-8766} --no-browser"]
