# Use an official lightweight Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables ensuring output is sent straight to terminal silently
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=10000

# Set the working directory in the container
WORKDIR /app

# Install critical system dependencies required for Web Compilers and AI ML models
# build-essential installs g++ natively
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ffmpeg \
    libsm6 \
    libxext6 \
    nodejs \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt /app/

# Upgrade pip securely
RUN pip install --upgrade pip

# Install heavily optimized AI backend configurations specifically without throwing cache bloat errors
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend directory architecture into the production container
COPY . /app/

# Expose the port so Render can physically interface with the socket connections
EXPOSE 10000

# Start Gunicorn server integrated flawlessly with socketio processing environments
CMD gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT app:app
