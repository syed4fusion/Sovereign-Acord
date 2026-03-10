#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import google.generativeai as genai

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}
DEFAULT_MODEL = "gemini-2.5-flash-lite-latest"
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
logger = logging.getLogger("gemini.extract")


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


def configure_genai():
    api_key = os.getenv("GENAI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GENAI_API_KEY or GOOGLE_API_KEY environment variable.")
    genai.configure(api_key=api_key)





def _collect_images(crops_dir: Path) -> List[Path]:
    out: List[Path] = []
    for p in sorted(crops_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            out.append(p)
    return out


def _chunked(items: List[Path], size: int) -> List[List[Path]]:
    if size <= 0:
        size = 1
    return [items[i : i + size] for i in range(0, len(items), size)]


def _build_batch_prompt(base_prompt: str, filenames: List[str]) -> str:
    base = (base_prompt or DEFAULT_PROMPT).strip()
    listed = "\n".join(f"- {name}" for name in filenames)
    return (
        f"{base}\n\n"
        "You are receiving multiple cropped images in this single request. For each image, "
        "extract the visible information and return a JSON array. Each array entry MUST contain:\n"
        '  - "file": the exact filename from the list below.\n'
        '  - "markdown": the Markdown extraction for that image.\n'
        "Keep the array order identical to the filename order provided below.\n"
        f"Filenames:\n{listed}\n"
        "Respond with JSON only."
    )


def _extract_one(image_path: Path, model, prompt: str, retries: int = 2) -> Dict[str, Any]:
    """Returns {'file', 'page', 'idx', 'extraction'} where extraction is a Markdown string."""
    entry: Dict[str, Any] = {"file": image_path.name}
    meta = parse_crop_filename(image_path)
    if meta:
        entry.update(meta)

    try:
        img_file = genai.upload_file(str(image_path))
        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                resp = model.generate_content([img_file, prompt])
                raw = (resp.text or "").strip()
                entry["extraction"] = raw
                return entry
            except Exception as exc:
                last_err = exc
                logger.warning(
                    "Gemini extraction attempt %d/%d failed for %s: %s",
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
        logger.error("Gemini extraction failed for %s: %s", image_path.name, exc)
        return entry


def _extract_batch(image_paths: List[Path], model, base_prompt: str, retries: int = 2) -> List[Dict[str, Any]]:
    filenames = [p.name for p in image_paths]
    prompt = _build_batch_prompt(base_prompt, filenames)
    uploads = [genai.upload_file(str(p)) for p in image_paths]
    meta_lookup = {p.name: parse_crop_filename(p) for p in image_paths}

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = model.generate_content([*uploads, prompt])
            raw = (resp.text or "").strip()
            if not raw:
                raise ValueError("Empty response from Gemini batch call")
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.debug("Batch raw response for %s: %s", filenames, raw)
                raise ValueError("Response was not valid JSON") from exc
            if not isinstance(data, list):
                raise ValueError("Batch response JSON is not a list")
            if len(data) != len(filenames):
                raise ValueError(
                    f"Expected {len(filenames)} entries but received {len(data)}"
                )

            entries: List[Dict[str, Any]] = []
            for expected, item in zip(filenames, data):
                if not isinstance(item, dict):
                    raise ValueError("Batch item is not an object")
                file_name = item.get("file") or expected
                markdown = (item.get("markdown") or item.get("extraction") or "").strip()
                entry: Dict[str, Any] = {"file": file_name, "extraction": markdown}
                meta = meta_lookup.get(file_name) or meta_lookup.get(expected)
                if meta:
                    entry.update(meta)
                entries.append(entry)
            return entries
        except Exception as exc:
            last_err = exc
            logger.warning(
                "Gemini batch extraction attempt %d/%d failed for %s: %s",
                attempt + 1,
                retries + 1,
                filenames,
                exc,
            )
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
    if last_err:
        raise last_err
    return []


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
    Extract data (as Markdown) from all crop images in `crops_dir`.

    Args:
        annotations_json: Source annotation JSON used to create the crops.
        crops_dir: Directory containing crop images.
        prompt: Optional override prompt text.
        model_name: Gemini model identifier.
        max_workers: Max threads for concurrent Gemini calls (default auto).
        max_output_tokens: Token budget for each Gemini Markdown extraction.
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

    # Configure Gemini (API key etc.)
    configure_genai()

    # Create the model directly (no client object)
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config={
            "temperature": 0.0,
            "max_output_tokens": max_output_tokens,
        },
    )

    prompt_text = (prompt or DEFAULT_PROMPT).strip()
    max_workers = max_workers or min(64, max(1, (os.cpu_count() or 4) * 2))

    results: List[Dict[str, Any]] = []
    order_map = {p.name: idx for idx, p in enumerate(images)}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {
            pool.submit(
                _extract_one,
                p,
                model,
                prompt_text,
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
        enrich = page_idx_map.get((int(page), int(idx))) if page and idx else {}
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
