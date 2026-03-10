import os
import shutil
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn

from default_cli import run_default_cli

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Sovereign ACORD Extraction API", description="API to extract data from ACORD PDFs")

# Environment/Configuration setup
COORDS_DIR = Path(os.getenv("ACORD_COORDS_DIR", "coords"))
CROPS_DIR = Path(os.getenv("ACORD_CROPS_DIR", "crops"))
EXTRACTION_MODEL = os.getenv("ACORD_EXTRACTION_MODEL", "gemini-2.5-flash-lite")
DETECT_MODEL = os.getenv("ACORD_DETECT_MODEL", "gemini-2.5-flash")
MAX_WORKERS = int(os.getenv("ACORD_MAX_WORKERS", "32"))

@app.post("/extract")
async def extract_acord(file: UploadFile = File(...)):
    """
    Upload an ACORD PDF file and receive extracted Markdown/JSON data.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    log.info(f"Received file for extraction: {file.filename}")
    temp_pdf_path = None
    
    try:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            shutil.copyfileobj(file.file, temp_pdf)
            temp_pdf_path = Path(temp_pdf.name)
    except Exception as e:
        log.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.")
    finally:
        file.file.close()

    try:
        # Run the extraction pipeline
        result = run_default_cli(
            pdf_path=temp_pdf_path,
            coords_dir=COORDS_DIR,
            model=EXTRACTION_MODEL,
            detect_model=DETECT_MODEL,
            max_output_tokens=4096,
            segment_max_output_tokens=3048,
            dpi=300,
            max_workers=MAX_WORKERS,
            segment_workers=MAX_WORKERS,
            prompt=None, 
        )
        
        # Cleanup temp file
        if temp_pdf_path and temp_pdf_path.exists():
            temp_pdf_path.unlink()
            
        return {"status": "success", "data": result}
        
    except Exception as e:
        log.error(f"Extraction failed: {e}")
        # Attempt cleanup on error
        if temp_pdf_path and temp_pdf_path.exists():
            temp_pdf_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Health check endpoint for RunPod/k8s"""
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    # Host on 0.0.0.0 to be accessible inside the RunPod container
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)
