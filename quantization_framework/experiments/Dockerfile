# Use the official PyTorch image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    python3-pip && \
    pip install --no-cache-dir numpy pandas matplotlib

# Set the working directory inside the container
WORKDIR /scratch/monacdan 

# Default command when the container starts
CMD ["/bin/bash"]
