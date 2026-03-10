import logging
import os
from pathlib import Path
from default_cli import run_default_cli  # Assuming default_cli.py is available in the same directory
from acord_extraction import run_pipeline  # Assuming acord_extraction.py is available in the same directory

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Toggle flags for features (use True/False to enable/disable features)
USE_CUSTOM_PROMPT = True
CUSTOM_PROMPT = "Extract the key data in a clear markdown table."
PRINT_MARKDOWN = True
SAVE_MARKDOWN_TO_FILE = True

# Set the input PDF path via environment variable ACORD_PDF_PATH
PDF_PATH_ENV = os.getenv("ACORD_PDF_PATH", "").strip()
PDF_PATH = Path(PDF_PATH_ENV) if PDF_PATH_ENV else None

# Optional configurations for the pipeline
COORDS_DIR = Path(r"coords")  # Change this to your actual coordinates directory
CROPS_DIR = Path(r"crops")  # Output crops directory
OUTPUT_MD_PATH = Path(r"output.md")  # Output file path for markdown extraction

def run_acord_extraction():
    # Log the start of the process
    if not PDF_PATH or not PDF_PATH.exists():
        raise FileNotFoundError("Set ACORD_PDF_PATH to an existing PDF file.")
    log.info(f"Starting ACORD extraction for {PDF_PATH.name}")

    try:
        # Run the extraction pipeline
        result = run_default_cli(
            pdf_path=PDF_PATH,
            coords_dir=COORDS_DIR,
            model="gemini-flash-lite-latest",
            detect_model=None,
            max_output_tokens=4096,
            segment_max_output_tokens=3048,
            dpi=300,
            max_workers=32,
            segment_workers=32,
            prompt=CUSTOM_PROMPT if USE_CUSTOM_PROMPT else None,
        )

        # Check and print results
        if PRINT_MARKDOWN:
            rendered_text = result.get("rendered_text", "")
            log.info("Rendered Markdown Output:\n" + rendered_text)

        # Optionally save the rendered output to a file
        if SAVE_MARKDOWN_TO_FILE and OUTPUT_MD_PATH:
            OUTPUT_MD_PATH.write_text(rendered_text, encoding="utf-8")
            log.info(f"Saved markdown to {OUTPUT_MD_PATH}")

        return result  # Return the result for further inspection or processing

    except Exception as e:
        log.error(f"An error occurred during ACORD extraction: {e}")
        raise  # Reraise the exception after logging

if __name__ == "__main__":
    result = run_acord_extraction()
    log.info("ACORD extraction completed successfully.")
