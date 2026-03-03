# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HOME=/home/user
ENV PATH="/home/user/.local/bin:${PATH}"

# Install system dependencies
# ffmpeg: required for audio processing in audio_processor.py
# libsndfile1: required for soundfile library
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
# We copy requirements first to leverage Docker cache
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the rest of the application code
COPY --chown=user . .

# Expose the port FastAPI will run on (7860 is the default for HF Spaces)
EXPOSE 7860

# Start the application using uvicorn
# Hugging Face Spaces expects the app to be on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
