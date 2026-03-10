#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import fitz  # PyMuPDF


def sanitize(text: str, fallback: str = "region") -> str:
    if not text:
        return fallback
    text = re.sub(r"[^\w\-]+", "-", text.strip())
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or fallback


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def choose_pdf_path(args_pdf: Optional[Path], json_data: Dict[str, Any], json_file: Path) -> Path:
    if args_pdf:
        return args_pdf
    src = (json_data or {}).get("source", {})
    name = src.get("name")
    if name:
        candidate = (json_file.parent / name).resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "PDF path not provided and could not be inferred from JSON 'source.name'. "
        "Pass --pdf /path/to/file.pdf"
    )


def rect_from_norm(norm: Dict[str, float], page_w: float, page_h: float) -> fitz.Rect:
    x0 = max(0.0, min(page_w, norm["x"] * page_w))
    y0 = max(0.0, min(page_h, norm["y"] * page_h))
    x1 = max(0.0, min(page_w, x0 + norm["w"] * page_w))
    y1 = max(0.0, min(page_h, y0 + norm["h"] * page_h))
    return fitz.Rect(x0, y0, x1, y1)


def rect_from_points_like_top_left(pts: Dict[str, float]) -> fitz.Rect:
    x0 = pts["x"]
    y0 = pts["y"]
    x1 = x0 + pts["w"]
    y1 = y0 + pts["h"]
    return fitz.Rect(x0, y0, x1, y1)


def maybe_flip_y(rect: fitz.Rect, page_h: float) -> fitz.Rect:
    h = rect.height
    y_top = page_h - (rect.y1)
    flipped = fitz.Rect(rect.x0, y_top, rect.x1, y_top + h)
    return flipped


def rect_in_bounds(r: fitz.Rect, page_w: float, page_h: float) -> bool:
    return (0 <= r.x0 < r.x1 <= page_w) and (0 <= r.y0 < r.y1 <= page_h)


def pick_rect_for_annotation(a: Dict[str, Any], page: fitz.Page) -> fitz.Rect:
    pw, ph = page.rect.width, page.rect.height
    pts = a.get("rect_pdf_points")
    if pts:
        r = rect_from_points_like_top_left(pts)
        if rect_in_bounds(r, pw, ph):
            return r
        r2 = maybe_flip_y(r, ph)
        if rect_in_bounds(r2, pw, ph):
            return r2
    norm = a.get("rect_norm")
    if not norm:
        raise ValueError("Annotation missing both rect_pdf_points and rect_norm.")
    return rect_from_norm(norm, pw, ph)


def save_clip(page: fitz.Page, rect: fitz.Rect, out_path: Path, dpi: int) -> None:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(out_path))


def crop_regions(
    annotations_json: Path,
    pdf_path: Optional[Path],
    out_dir: Path,
    dpi: int = 300,
    prefix: str = "",
    doc: Optional[fitz.Document] = None,
) -> Dict[str, Any]:
    """
    Import-friendly entry point. Returns metadata with saved files.
    Optionally reuse an open PyMuPDF document to avoid reloading per call.
    """
    data = load_json(annotations_json)
    pdf = choose_pdf_path(pdf_path, data, annotations_json)
    if not pdf.exists():
        raise FileNotFoundError(f"PDF not found: {pdf}")

    annotations = data.get("annotations", [])
    if not annotations:
        return {"pdf": str(pdf), "out_dir": str(out_dir), "saved": [], "count": 0}

    close_doc = False
    if doc is None:
        doc = fitz.open(str(pdf))
        close_doc = True

    by_page: Dict[int, list] = {}
    for a in annotations:
        p = int(a.get("page", 1))
        by_page.setdefault(p, []).append(a)

    total_saved = 0
    saved_files: List[str] = []
    base = sanitize(prefix) if prefix else sanitize(pdf.stem)

    for page_num, anns in sorted(by_page.items()):
        if page_num < 1 or page_num > len(doc):
            continue
        page = doc[page_num - 1]
        pw, ph = page.rect.width, page.rect.height

        for idx, a in enumerate(anns, start=1):
            try:
                rect = pick_rect_for_annotation(a, page)
            except Exception:
                continue

            rect = fitz.Rect(
                max(0, rect.x0),
                max(0, rect.y0),
                min(pw, rect.x1),
                min(ph, rect.y1),
            )
            if rect.is_empty or rect.width < 1 or rect.height < 1:
                continue

            label = sanitize(a.get("label", "region"))
            out_name = f"{base}_p{page_num:03d}_{label}_{idx:03d}.png"
            out_path = out_dir / out_name
            save_clip(page, rect, out_path, dpi=dpi)
            saved_files.append(str(out_path))
            total_saved += 1

    if close_doc:
        doc.close()
    return {
        "pdf": str(pdf),
        "out_dir": str(out_dir),
        "saved": saved_files,
        "count": total_saved,
        "annotations_json": str(annotations_json),
    }


# ─────────────────────────────
# CLI still supported
# ─────────────────────────────
def _parse_args():
    ap = argparse.ArgumentParser(description="Crop PDF regions from annotator JSON and save as PNGs.")
    ap.add_argument("--json", required=True, type=Path)
    ap.add_argument("--pdf", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--prefix", type=str, default="")
    return ap.parse_args()


def main():
    args = _parse_args()
    result = crop_regions(args.json, args.pdf, args.out, dpi=args.dpi, prefix=args.prefix)
    for f in result["saved"]:
        print(f"Saved {f}")
    print(f"Done. Saved {result['count']} crops to: {Path(result['out_dir']).resolve()}")


if __name__ == "__main__":
    main()
