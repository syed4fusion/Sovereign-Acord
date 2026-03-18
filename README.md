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
Go to your RunPod dashboard, find your running pod, and click **Connect**. 
Click the **HTTP Service (Port 8000)** button. This will open your unique API URL, which looks like:
`https://<pod-id>-8000.proxy.runpod.net/docs`

*(The `pod-id` is the unique code for your instance, e.g., `abc123def456`)*

### 2. Using cURL
Copy the base URL from the Step above and use it in your cURL command:
```bash
curl -X 'POST' \
  'https://YOUR-POD-ID-8000.proxy.runpod.net/extract' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@path/to/your/acord_form.pdf;type=application/pdf'
```

---

## 🛠️ Troubleshooting

### "Address already in use" (OSError 98)
If you get a port conflict error and `fuser` is missing, use this command to clear the ports:
```bash
ps aux | grep -e vllm -e uvicorn | awk '{print $2}' | xargs kill -9 2>/dev/null
```
Then run `bash start.sh` again.

### vLLM Startup Time
The `dots.ocr` model is large. It may take 2-3 minutes to load into VRAM. If the API returns a connection error immediately, wait another minute and try again.

---

## ✅ How to Verify Success

Once the server is running, you can verify it using one of the following methods:

### 1. Interactive API Docs (Recommended)
This is the easiest way to test visually:
1. Go to your RunPod dashboard.
2. Find your Pod and click **Connect**.
3. Select **HTTP Service (Port 8000)**.
4. Add `/docs` to the end of the URL in your browser (e.g., `https://<pod-id>-8000.proxy.runpod.net/docs`).
5. Locate the **POST /extract** endpoint, click **Try it out**, upload an ACORD PDF, and click **Execute**.

### 2. Using cURL
Run this command from your terminal (replacing the URL and file path):
```bash
curl -X 'POST' \
  'https://YOUR-POD-ID-8000.proxy.runpod.net/extract' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/path/to/your/acord_form.pdf;type=application/pdf'
```

### 3. Monitoring Backend Logs
While the extraction is running, watch your terminal logs. You should see entries like this confirming OCR activity:
- `INFO: dots.extract - Dots.ocr extraction attempt ...`
- `INFO: vllm - Processing request ...`

---

## 📂 Project Structure
- `api.py`: The FastAPI server.
- `start.sh`: Manages both `vLLM` (on port 8005) and the `FastAPI Gateway` (on port 8000).
- `dots_extract.py`: Handles OCR chunks using the local vLLM instance.
- `coords/`: Essential coordinate mapping JSONs for ACORD forms.
- `requirements.txt`: Project dependencies.
