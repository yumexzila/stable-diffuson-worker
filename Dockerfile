# Dockerfile for stable-diffusion-worker

# Stage 1: Build Image - Use a CUDA-enabled base image for GPU support
# This is critical! Choose a base image that matches your CUDA/cuDNN requirements.
# nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 is a good general choice.
# You might need to adjust based on specific PyTorch/Diffusers requirements.
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory inside the container
WORKDIR /app

# Install Python and pip (if not already in the base image, though this CUDA image should have it)
# Ensure apt-get is updated and python3-pip is installed
RUN apt-get update && apt-get install -y python3 python3-pip

# Copy requirements.txt and install Python dependencies
# We copy requirements.txt first to leverage Docker's build cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code into the container
COPY . .

# Expose the port your FastAPI application will run on
# This is the CONTAINER PORT (8000 for Stable Diffusion worker)
EXPOSE 8000

# Define environment variable for the API key (will be set by Salad.io during deployment)
# This just declares it. The actual value comes from Salad.io.
ENV API_SECRET_KEY=

# Command to run your FastAPI application when the container starts
# Make sure to specify the correct host and port
# 'main:app' means run the 'app' object from 'main.py'
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
