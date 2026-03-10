# Sovereign ACORD Extraction API

A FastAPI-based gateway for automated ACORD PDF field extraction, powered by **Dots.ocr-1.5** (via vLLM) and **Gemini** (for auto-detection).

## 🚀 RunPod Deployment Guide

### 1. Launch a GPU Pod
- **GPU Recommended**: RTX 3090, 4090, or A6000 (minimum 24GB VRAM).
- **Network Configuration**: 
  - Expose HTTP Ports: `8000, 8888`
  - Expose TCP Ports: `22` (SSH)

### 2. Connect & Setup
Open the **Web Terminal** and run the following:

```bash
# 1. Clone the repository (Replace with your GitHub Personal Access Token)
git clone https://<YOUR_GITHUB_TOKEN>@github.com/syed4fusion/Sovereign-Acord.git
cd Sovereign-Acord

# 2. Setup environment variables
export GENAI_API_KEY="your_gemini_api_key_here"

# 3. Create virtual environment & install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Start the application
bash start.sh
```

---

## 🛠️ Testing the API

Once you see **"vLLM is ready!"** in your terminal:

### 1. Interactive Docs (Swagger)
Go to your RunPod dashboard, click **Connect**, and open the **HTTP Service (Port 8000)**. 
Add `/docs` to the end of the URL:
`https://<pod-id>-8000.proxy.runpod.net/docs`

### 2. Using cURL
```bash
curl -X 'POST' \
  'https://<pod-id>-8000.proxy.runpod.net/extract' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@path/to/your/acord_form.pdf;type=application/pdf'
```

---

## 📂 Project Structure
- `api.py`: The FastAPI server.
- `start.sh`: Manages both `vLLM` (on port 8001) and the `FastAPI Gateway` (on port 8000).
- `dots_extract.py`: Handles OCR chunks using the local vLLM instance.
- `coords/`: Essential coordinate mapping JSONs for ACORD forms.
- `requirements.txt`: Project dependencies.
