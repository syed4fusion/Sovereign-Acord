#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import base64
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
# Modified to default to port 8005 for internal vLLM on pod
DEFAULT_API_BASE = "http://localhost:8005/v1" 
DEFAULT_MODEL = "rednote-hilab/dots.ocr"
INTER_REQUEST_SLEEP = 0.0  # bump if you hit rate limits

FNAME_RE = re.compile(
    r"""_p(?P<page>\d{3})_(?P<label_sanitized>.+)_(?P<idx>\d{3})\.(?:png|jpg|jpeg|webp|tif|tiff|bmp)$""",
    re.IGNORECASE,
)

DEFAULT_PROMPT = (
    "You will receive a cropped image of an insurance form (e.g., ACORD). "
    "Extract only the information visible within this region. "
    "There will be weirdly formatted tables. "
    "keep the paragraph and writing regions small if nothing is present. "
    "Return the result as clean Markdown. "
    "Do not include any extra commentary beyond the extracted content."
)
logger = logging.getLogger("dots.extract")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_crop_filename(p: Path):
    m = FNAME_RE.search(p.name)
    if not m:
        return None
    return {
        "page": int(m.group("page")),
        "label_sanitized": m.group("label_sanitized"),
        "idx": int(m.group("idx")),
    }


def load_annotations(json_path: Path) -> Dict[str, Any]:
    return json.loads(json_path.read_text(encoding="utf-8"))


def build_page_index_map(annotations_json: Dict[str, Any]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    by_page: Dict[int, List[Dict[str, Any]]] = {}
    for a in annotations_json.get("annotations", []):
        page = int(a.get("page", 1))
        by_page.setdefault(page, []).append(a)
    mapping: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for page, anns in by_page.items():
        for idx, a in enumerate(anns, start=1):
            mapping[(page, idx)] = {
                "label": a.get("label", "region"),
                "rect_px": a.get("rect_px"),
                "rect_norm": a.get("rect_norm"),
                "rect_pdf_points": a.get("rect_pdf_points"),
            }
    return mapping


def _collect_images(crops_dir: Path) -> List[Path]:
    out: List[Path] = []
    for p in sorted(crops_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            out.append(p)
    return out


def _encode_image(image_path: Path) -> str:
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def _get_api_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    """Get HTTP headers for the API request"""
    headers = {
        "Content-Type": "application/json"
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _extract_one(image_path: Path, api_base: str, model_name: str, prompt: str, max_tokens: int, api_key: Optional[str] = None, retries: int = 2) -> Dict[str, Any]:
    """Returns {'file', 'page', 'idx', 'extraction'} where extraction is a Markdown string."""
    entry: Dict[str, Any] = {"file": image_path.name}
    meta = parse_crop_filename(image_path)
    if meta:
        entry.update(meta)

    # Note: vLLM implements the OpenAI Chat Completion API 
    endpoint = f"{api_base.rstrip('/')}/chat/completions"
    
    try:
        base64_image = _encode_image(image_path)
        # Infer the mime type or default to image/jpeg
        mime_type = f"image/{image_path.suffix.lower().lstrip('.')}"
        if mime_type == "image/jpg": mime_type = "image/jpeg"
        
        image_url = f"data:{mime_type};base64,{base64_image}"

        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0
        }
        
        headers = _get_api_headers(api_key)

        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                response = requests.post(endpoint, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                response_json = response.json()
                
                # Parse OpenAI compatible response structure
                if "choices" in response_json and len(response_json["choices"]) > 0:
                    raw = response_json["choices"][0].get("message", {}).get("content", "").strip()
                    entry["extraction"] = raw
                else:
                    raise ValueError(f"Unexpected response format: {response_json}")
                
                return entry
            except Exception as exc:
                last_err = exc
                logger.warning(
                    "Dots.ocr extraction attempt %d/%d failed for %s: %s",
                    attempt + 1,
                    retries + 1,
                    image_path.name,
                    exc,
                )
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise
        if last_err:
            raise last_err
    except Exception as exc:
        entry["error"] = f"{type(exc).__name__}: {exc}"
        entry["extraction"] = ""
        logger.error("Dots.ocr extraction failed for %s: %s", image_path.name, exc)
        return entry

# ---------------------------------------------------------------------------
# Importable entry point
# ---------------------------------------------------------------------------

def extract_crops(
    annotations_json: Path,
    crops_dir: Path,
    prompt: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
    max_workers: Optional[int] = None,
    max_output_tokens: int = 2048,
) -> Dict[str, Any]:
    """
    Extract data (as Markdown) from all crop images in `crops_dir` using Dots.ocr via vLLM.

    Args:
        annotations_json: Source annotation JSON used to create the crops.
        crops_dir: Directory containing crop images.
        prompt: Optional override prompt text.
        model_name: Dots model identifier (e.g. rednote-hilab/dots.ocr-1.5).
        max_workers: Max threads for concurrent API calls (default auto).
        max_output_tokens: Token budget for each markdown extraction.
    """
    if not crops_dir.exists():
        raise FileNotFoundError(f"Crops dir not found: {crops_dir}")

    annotations = load_annotations(annotations_json)
    page_idx_map = build_page_index_map(annotations)

    images = _collect_images(crops_dir)
    if not images:
        return {
            "annotations_json": str(annotations_json),
            "crops_dir": str(crops_dir),
            "model": model_name,
            "files": [],
        }

    api_base = os.getenv("DOTS_OCR_API_BASE", DEFAULT_API_BASE)
    api_key = os.getenv("DOTS_OCR_API_KEY", "") # Some cloud providers need an API Key on top

    prompt_text = (prompt or DEFAULT_PROMPT).strip()
    max_workers = max_workers or min(64, max(1, (os.cpu_count() or 4) * 2))

    results: List[Dict[str, Any]] = []
    order_map = {p.name: idx for idx, p in enumerate(images)}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {
            pool.submit(
                _extract_one,
                p,
                api_base,
                model_name,
                prompt_text,
                max_output_tokens,
                api_key
            ): p
            for p in images
        }
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            if INTER_REQUEST_SLEEP > 0:
                time.sleep(INTER_REQUEST_SLEEP)

    # ensure deterministic ordering matching the original image list
    results.sort(key=lambda item: order_map.get(item.get("file", ""), 1_000_000))

    merged: List[Dict[str, Any]] = []
    for item in results:
        page = item.get("page")
        idx = item.get("idx")
        enrich = page_idx_map.get((int(page), int(idx)), {}) if page and idx else {}
        merged.append(
            {
                **item,
                "label_original": enrich.get("label"),
                "rect_px": enrich.get("rect_px"),
                "rect_norm": enrich.get("rect_norm"),
                "rect_pdf_points": enrich.get("rect_pdf_points"),
            }
        )

    return {
        "annotations_json": str(annotations_json),
        "crops_dir": str(crops_dir),
        "model": model_name,
        "files": merged,
    }

