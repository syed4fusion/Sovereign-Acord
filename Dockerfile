# Use RunPod's standard PyTorch image for GPU support
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Setup workspace
WORKDIR /workspace

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Ensure start.sh is executable
RUN chmod +x start.sh

# Expose ports for UI (8888) and API (8000)
EXPOSE 8888 8000 8001

# Entrypoint manages both vLLM and our FastAPI app
ENTRYPOINT ["./start.sh"]
