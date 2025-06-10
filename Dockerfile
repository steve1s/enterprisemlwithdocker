# Use official Python image with system-level dependencies for TensorFlow and OpenCV
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender-dev \
    libxext6 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements-docker.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements-docker.txt

# Copy the rest of the app
COPY . .

# Expose port
EXPOSE 5000

# Set environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=production

# Download model if not present (optional, comment out if you want to mount it)
# RUN wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz \
#     && tar -xzf ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz \
#     && mv ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model ./saved_model

# Start the Flask app
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"]
