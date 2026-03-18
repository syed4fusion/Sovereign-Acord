#!/bin/bash

# Configuration
MODEL="rednote-hilab/dots.ocr"
API_PORT=${PORT:-8000}
VLLM_PORT=8005

echo "--- Starting vLLM model server on port $VLLM_PORT ---"
# Launch vLLM in the background. 
# Using --port 8005 so it doesn't conflict with our FastAPI's 8000
# Adjust gpu-memory-utilization if needed, dots.ocr-1.5 fits in most 24GB+ GPUs
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$VLLM_PORT" \
    --host 0.0.0.0 \
    --trust-remote-code \
    --gpu-memory-utilization 0.8 &

# Wait for vLLM to be ready
echo "--- Waiting for vLLM to initialize (can take 1-3 minutes)... ---"
until curl -s http://localhost:$VLLM_PORT/v1/models > /dev/null; do
    sleep 5
done
echo "--- vLLM is ready! Starting FastAPI Gateway on port $API_PORT ---"

# Point the app to the internal vLLM port
export DOTS_OCR_API_BASE="http://localhost:$VLLM_PORT/v1"

# Start FastAPI Gateway
python3 -m uvicorn api:app --host 0.0.0.0 --port "$API_PORT"
