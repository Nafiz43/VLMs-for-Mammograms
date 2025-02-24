# Use a minimal base OS image with Python pre-installed
FROM python:3.11-slim

# Set a working directory in the container
WORKDIR /app

# Update the package manager and install pip (if not already installed)
RUN apt-get update && apt-get install -y \
    build-essential \
    && pip install --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

# Install the required Python packages
RUN pip install langchain click pandas langchain_community langchain-ollama
RUN apt-get update && apt-get install -y curl
RUN apt-get install -y pciutils lshw
RUN curl -fsSL https://ollama.com/install.sh | sh

COPY . /app

# Default command (optional, adjusts based on your use case)
CMD ["python3"]