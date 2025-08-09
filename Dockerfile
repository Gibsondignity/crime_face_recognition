# Use official Python base image
FROM python:3.8-slim

# Install dependencies for dlib, face_recognition, and opencv
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libboost-all-dev \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    libv4l-dev \
    v4l-utils \
    git \
    wget \
    && apt-get clean

# Create working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the Django app
COPY . .

# Expose port
EXPOSE 8000

# Run Django server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
