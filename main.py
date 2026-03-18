#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py

Thin CLI around the ACORD extraction pipeline + optional auto-detect + PDF splitting,
now with structured, pretty logs.

Modes
-----
1) Single-run (one coords JSON)
2) Segments mode (known ACORD page spans)
3) Auto-detect ACORD spans with Gemini, then extract (+ optional splitting)

Auth
----
- Set GENAI_API_KEY or GOOGLE_API_KEY in your environment for Gemini.
"""

from __future__ import annotations

import argparse
import json
import os
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
try:
    from typing_extensions import TypedDict
except ImportError:
    from typing import TypedDict

# pipeline entry points
from acord_extraction import (
    run_pipeline,
    run_segments_pipeline,
    SegmentSpec,
)

# --- detection (Gemini) ---
import google.generativeai as genai  # uses GENAI_API_KEY/GOOGLE_API_KEY

# --- splitting (PDF) ---
try:
    from pypdf import PdfReader, PdfWriter
except Exception:
    from PyPDF2 import PdfReader, PdfWriter  # type: ignore


# =============================== logging setup ===============================

def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """
    Configure root logging. Uses RichHandler if available for pretty console logs.
    """
    numeric = getattr(logging, (level or "INFO").upper(), logging.INFO)

    handlers: list[logging.Handler] = []
    # Console handler (prefer Rich)
    try:
        from rich.logging import RichHandler  # type: ignore
        handlers.append(RichHandler(rich_tracebacks=True, show_time=True, show_level=True, show_path=False))
        console_fmt = "%(message)s"
        date_fmt = "[%X]"
        logging.basicConfig(level=numeric, format=console_fmt, datefmt=date_fmt, handlers=handlers)
    except Exception:
        console_fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
        logging.basicConfig(level=numeric, format=console_fmt)

    # Optional file handler
    if log_file:
        fh = logging.FileHandler(str(log_file), encoding="utf-8")
        fh.setLevel(numeric)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logging.getLogger().addHandler(fh)

    # Quiet down super chatty libs (adjust as needed)
    for noisy in ("google.generativeai", "httpx", "urllib3"):
        logging.getLogger(noisy).setLevel(max(numeric, logging.WARNING))


log = logging.getLogger("main")

# Dedicated logger for ACORD identification
acord_log = logging.getLogger("acord")


# ========================= Gemini-based ACORD detection =========================

class DetectedForm(TypedDict, total=False):
    form_number: str
    form_title: str
    edition: str
    start_page: int
    end_page: int
    confidence: float
    raw_markings: List[str]
    potential_coord_mapping: str  # e.g., "bp_acord_80__201309__us-national.json"


DETECTION_PROMPT = """
You are given a PDF that may contain one or more ACORD® forms.
Return a JSON array where each item has:
- form_number (e.g., "ACORD 25", "ACORD 28", "ACORD 80", etc.)
- form_title (as printed)
- edition (e.g., "2016/03" or "(2018/09)" — include slashes/parentheses if shown)
- start_page (1-based, inclusive)
- end_page   (1-based, inclusive)
- confidence in [0,1]
- raw_markings: short strings that justified your identification (e.g., "ACORD 25 (2016/03)")
- potential_coord_mapping: a filename for the best-fit coords JSON (e.g., "bp_acord_25__201603__us-national.json")

Rules:
- Use the PDF's own page numbering starting at 1.
- If a form spans multiple pages, return one item with start_page and end_page covering the span.
- Only include pages that are part of ACORD forms.
Return ONLY JSON.
""".strip()


def _configure_genai():
    api_key = os.getenv("GENAI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GENAI_API_KEY or GOOGLE_API_KEY environment variable.")
    genai.configure(api_key=api_key)
    logging.getLogger(__name__).debug("Gemini client configured using environment key.")


def auto_detect_acord_spans(
    pdf_path: Path,
    model_name: str = "gemini-2.5-flash",
    max_output_tokens: int = 2048,
    timeout_sec: float = 120.0,
) -> List[DetectedForm]:
    """
    Uses Gemini Files API + structured output to detect ACORD forms and page spans.
    """
    logger = logging.getLogger("detect")
    t0 = time.perf_counter()
    logger.info("Auto-detect: starting (model=%s)", model_name)
    logger.debug("Auto-detect params: pdf=%s, max_output_tokens=%d, timeout=%.1fs",
                 pdf_path, max_output_tokens, timeout_sec)

    _configure_genai()

    # Upload the PDF once and reuse it in the prompt
    logger.debug("Uploading PDF to Gemini Files API: %s", pdf_path)
    uploaded = genai.upload_file(str(pdf_path), mime_type="application/pdf")
    logger.debug("Upload complete: file=%s", getattr(uploaded, "name", "<unknown>"))

    # Model with JSON-mode (structured output)
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=genai.GenerationConfig(
            temperature=0.0,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
            response_schema=list[DetectedForm],
        ),
    )

    try:
        logger.info("Auto-detect: generating structured JSON from Gemini…")
        resp = model.generate_content(
            contents=[uploaded, DETECTION_PROMPT],
            request_options={"timeout": float(timeout_sec)},
        )
        text = (resp.text or "").strip()
        if not text:
            logger.warning("Auto-detect: empty response from model.")
            return []
        items: List[DetectedForm] = json.loads(text)
    except Exception:
        logger.exception("Auto-detect: Gemini request failed.")
        raise

    # basic sanitation
    cleaned: List[DetectedForm] = []
    for it in items:
        try:
            sp = max(1, int(it.get("start_page", 1)))
            ep = max(sp, int(it.get("end_page", sp)))
            conf = float(it.get("confidence", 0.0))
            it["start_page"] = sp
            it["end_page"] = ep
            it["confidence"] = max(0.0, min(1.0, conf))
            cleaned.append(it)
        except Exception:
            logger.debug("Auto-detect: skipped malformed item: %r", it)

    cleaned.sort(key=lambda x: (x["start_page"], x["end_page"]))
    dt = time.perf_counter() - t0
    logger.info("Auto-detect: %d form(s) detected in %.2fs", len(cleaned), dt)
    for i, c in enumerate(cleaned, 1):
        logger.debug("  %02d) %s | %s | %s | pages %d-%d | conf=%.2f | hints=%s",
                     i, c.get("form_number"), c.get("form_title"), c.get("edition"),
                     c["start_page"], c["end_page"], c.get("confidence", 0.0),
                     ", ".join(c.get("raw_markings", [])[:3]))
    return cleaned


# =============================== PDF splitting ===============================

@dataclass
class SplitSpec:
    label: str      # e.g., "ACORD_25"
    start_page: int # 1-based inclusive
    end_page: int   # 1-based inclusive


def split_pdf_by_spans(
    pdf_path: Path,
    spans: List[SplitSpec],
    out_dir: Path,
    pattern: str = "{idx:02d}_{label}_p{start}-{end}.pdf",
    overwrite: bool = False,
) -> List[str]:
    """
    Write one PDF per span. Returns list of written file paths.
    """
    logger = logging.getLogger("split")
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(str(pdf_path))
    written: List[str] = []
    logger.info("Splitting PDF: %s → %d span(s) → %s", pdf_path, len(spans), out_dir)

    for i, s in enumerate(spans, start=1):
        start_idx = s.start_page - 1  # pypdf is 0-based
        end_idx = s.end_page - 1
        writer = PdfWriter()
        for p in range(start_idx, end_idx + 1):
            writer.add_page(reader.pages[p])
        fname = pattern.format(
            idx=i, label=s.label, start=s.start_page, end=s.end_page
        )
        out_path = out_dir / fname
        if out_path.exists() and not overwrite:
            logger.error("Split target exists and overwrite=False: %s", out_path)
            raise FileExistsError(f"{out_path} exists (use overwrite=True to replace).")
        with open(out_path, "wb") as f:
            writer.write(f)
        written.append(str(out_path))
        logger.info("  wrote: %s", out_path)

    dt = time.perf_counter() - t0
    logger.info("Splitting complete: %d file(s) in %.2fs", len(written), dt)
    return written


# =============================== Validation & logging helpers ===============================

def _pdf_num_pages(pdf_path: Path) -> int:
    reader = PdfReader(str(pdf_path))
    return len(reader.pages)


@dataclass
class PageFix:
    form_number: str
    original: tuple[int, int]
    fixed: tuple[int, int]
    reason: str


def _validate_and_normalize_segments(pdf_path: Path,
                                     segments: List[SegmentSpec]) -> tuple[List[SegmentSpec], List[PageFix]]:
    """
    Ensure 1-based pages are within [1..N] and start_page <= end_page.
    Returns possibly-adjusted segments and a list of fixes applied.
    """
    fixes: List[PageFix] = []
    n = _pdf_num_pages(pdf_path)
    normalized: List[SegmentSpec] = []

    for s in segments:
        orig = (s.start_page, s.end_page)
        sp = int(s.start_page)
        ep = int(s.end_page)

        # Fix swapped range
        if sp > ep:
            sp, ep = ep, sp
            fixes.append(PageFix(s.form_number, orig, (sp, ep), "start_page> end_page (swapped)"))

        # Clamp to PDF bounds
        new_sp = max(1, min(sp, n))
        new_ep = max(1, min(ep, n))

        if (new_sp, new_ep) != (sp, ep):
            fixes.append(PageFix(s.form_number, (sp, ep), (new_sp, new_ep),
                                 f"clamped to PDF bounds [1..{n}]"))
            sp, ep = new_sp, new_ep

        normalized.append(SegmentSpec(
            form_number=s.form_number,
            start_page=sp,
            end_page=ep,
            coords_json=s.coords_json,
            edition=s.edition,
            jurisdiction=s.jurisdiction,
        ))

    return normalized, fixes


def log_acord_identities(items: List[DetectedForm],
                         source: str = "auto-detect",
                         pdf: Optional[Path] = None,
                         coords_dir: Optional[Path] = None) -> None:
    if pdf:
        acord_log.info("ACORD identification (%s): %d form(s) in %s",
                       source, len(items), pdf)
    else:
        acord_log.info("ACORD identification (%s): %d form(s)", source, len(items))
    for it in items:
        pcm = it.get("potential_coord_mapping", "")
        acord_log.info(
            'ACORD identified | form_number="%s" title="%s" edition="%s" pages=%d-%d '
            'conf=%.2f source=%s coord_hint="%s"',
            it.get("form_number", "?"),
            it.get("form_title", "?"),
            it.get("edition", "?"),
            int(it.get("start_page", 0)),
            int(it.get("end_page", 0)),
            float(it.get("confidence", 0.0)),
            source,
            pcm or "",
        )
        acord_log.debug("ACORD identified (json) %s", json.dumps(it, ensure_ascii=False))


def _resolve_coords_path(coords_dir: Path, hint: Optional[str]) -> Optional[Path]:
    """
    Turn a model-provided filename (relative or absolute) into an absolute path if it exists.
    Tries a couple of simple fallbacks (e.g., stripping a 'bp_' prefix).
    """
    if not hint:
        return None
    candidates = []
    p = Path(hint)
    candidates.append(p if p.is_absolute() else coords_dir / p)

    # Simple fallback: strip a leading 'bp_' if present
    name = p.name
    if name.startswith("bp_"):
        candidates.append(coords_dir / name[3:])

    # De-dup and check
    seen = set()
    for c in candidates:
        try:
            c = c.expanduser().resolve()
        except Exception:
            continue
        if c in seen:
            continue
        seen.add(c)
        if c.exists():
            return c
    return None


def _write_detected_log(path: Path, items: List[DetectedForm]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suf = path.suffix.lower()
    if suf in (".jsonl", ".ndjson", ".jsonlines"):
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
    acord_log.info("Wrote detected ACORD log → %s", path)


def _write_missing_coords(path: Path, missing: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"missing_coord_mappings": missing}, f, ensure_ascii=False, indent=2)
    else:
        with open(path, "w", encoding="utf-8") as f:
            for m in missing:
                f.write(m + "\n")
    acord_log.info("Wrote missing coord mappings → %s", path)


# ================================ CLI parsing ================================

def _parse_args():
    ap = argparse.ArgumentParser(
        description="PDF → ACORD extraction via crops + optional auto-detect + PDF splitting."
    )
    # Common
    ap.add_argument("--pdf", type=Path, required=True,
                    help="Input master PDF (required in all modes)")
    ap.add_argument("--coords-dir", required=True, type=Path,
                    help="Folder containing ACORD coordinate JSONs")
    ap.add_argument("--crops-dir", required=True, type=Path,
                    help="Output folder for crops (used in all modes)")
    ap.add_argument("--dpi", type=int, default=144, help="DPI for crops")
    ap.add_argument("--model", type=str, default="gemini-2.5-flash-lite",
                    help="Gemini model (e.g., gemini-2.5-flash)")
    ap.add_argument("--detect-model", type=str, default="gemini-2.5-flash",
                    help="Gemini model to use for ACORD auto-detection")
    ap.add_argument("--max-workers", type=int, default=64,
                    help="Threads for extraction (default: auto)")
    ap.add_argument("--prompt", type=str,
                    help="Single prompt to send to Gemini (text/MD output)")
    ap.add_argument("--prompt-file", type=Path,
                    help="Load prompt from file")
    ap.add_argument("--out", type=Path,
                    help="Write the full JSON payload to this path (optional)")
    ap.add_argument("--render-out", type=Path,
                    help="Write only the ordered extraction text/MD here (optional)")
    ap.add_argument("--render-format", type=str, choices=["md", "text"], default="md",
                    help="Whether to render as Markdown or plain text")
    ap.add_argument("--keep-crops", action="store_true",
                    help="Keep the cropped images instead of deleting them (default: delete)")

    # removed gemini_batch_size
    # help removed
    ap.add_argument("--segment-workers", type=int, default=3,
                    help="Parallel workers for segment processing (default auto).")

    # Logging
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help="Console log level (default: INFO)")
    ap.add_argument("--log-file", type=Path,
                    help="Optional log file path (in addition to console).")
    ap.add_argument("--detected-log", type=Path,
                    help="Write identified ACORD forms (.json array or .jsonl/ndjson) to this path.")
    ap.add_argument("--missing-coords-out", type=Path,
                    help="Write list of ACORD form_numbers with missing coords_json to this path (txt or json).")

    # Single-run specific
    ap.add_argument("--coords-json", type=Path,
                    help="(single-run) Explicit coordinates JSON (overrides resolver)")
    ap.add_argument("--prefix", type=str, default="",
                    help="(single-run) Optional filename prefix for crops")

    # Segments mode
    ap.add_argument("--segments", type=Path,
                    help='JSON file describing segments: '
                         '[{"form_number":"ACORD 25","start_page":1,"end_page":1,'
                         '"coords_json":"coords/acord_25__201603__us-national.json"}, ...]')
    ap.add_argument("--default-edition", type=str,
                    help='Edition hint when segments omit coords_json (e.g., "2016/11")')
    ap.add_argument("--default-jurisdiction", type=str,
                    help='Jurisdiction hint (e.g., "US-National", "IL", "FL")')

    # Auto-detect mode
    ap.add_argument("--auto-detect", action="store_true",
                    help="Detect ACORD forms + page spans with Gemini before extraction.")

    # Splitting
    ap.add_argument("--split-out", type=Path,
                    help="If provided, write per-form PDFs to this directory.")
    ap.add_argument("--split-pattern", type=str,
                    default="{idx:02d}_{label}_p{start}-{end}.pdf",
                    help="Filename pattern for split PDFs (see placeholders).")
    ap.add_argument("--split-overwrite", action="store_true",
                    help="Overwrite existing split PDFs if present.")

    return ap.parse_args()


# ================================== main ===================================

def _main_cli():
    args = _parse_args()
    setup_logging(level=args.log_level, log_file=args.log_file)
    # Ensure 'acord' logger respects CLI level too
    logging.getLogger("acord").setLevel(getattr(logging, (args.log_level or "INFO").upper(), logging.INFO))

    log.info("Starting main (mode=%s)", "auto-detect" if args.auto_detect else ("segments" if args.segments else "single-run"))
    log.debug("Args: pdf=%s, coords_dir=%s, crops_dir=%s, dpi=%d, model=%s, out=%s, render_out=%s, format=%s, keep_crops=%s",
              args.pdf, args.coords_dir, args.crops_dir, args.dpi, args.model, args.out, args.render_out, args.render_format, args.keep_crops)

    # --- AUTO-DETECT path ---
    if args.auto_detect:
        try:
            detected = auto_detect_acord_spans(
                pdf_path=args.pdf,
                model_name=args.detect_model,
            )
        except Exception:
            log.exception("Auto-detect failed.")
            raise

        if not detected:
            log.error("No ACORD forms detected.")
            raise SystemExit(2)

        # Log and optionally persist the detected list
        log_acord_identities(detected, source="auto-detect", pdf=args.pdf, coords_dir=args.coords_dir)
        if args.detected_log:
            _write_detected_log(args.detected_log, detected)

        # Build SegmentSpec list for extraction using potential_coord_mapping
        segments: List[SegmentSpec] = []
        missing_coords: List[str] = []

        for d in detected:
            coord_path = _resolve_coords_path(args.coords_dir, d.get("potential_coord_mapping"))
            if coord_path:
                acord_log.info('Mapping %s → coords_json="%s"', d.get("form_number", "?"), coord_path)
            else:
                name = d.get("form_number", "?")
                hint = d.get("potential_coord_mapping")
                acord_log.warning('No coords_json found for %s using hint "%s". Will rely on pipeline resolver.',
                                  name, hint)
                missing_coords.append(name)

            segments.append(SegmentSpec(
                form_number=d["form_number"],
                start_page=d["start_page"],
                end_page=d["end_page"],
                coords_json=coord_path,             # resolved absolute path or None
                edition=d.get("edition"),
                jurisdiction=None,                  # can't infer reliably from PDF
            ))

        # Ensure page ranges are valid and within the PDF
        segments, page_fixes = _validate_and_normalize_segments(args.pdf, segments)
        for fx in page_fixes:
            log.warning('Page fix for %s: %s → %s (%s)',
                        fx.form_number, fx.original, fx.fixed, fx.reason)

        if missing_coords:
            acord_log.info("Missing coord mappings for %d ACORD(s): %s",
                           len(missing_coords), ", ".join(sorted(set(missing_coords))))

        log.info("Running extraction for %d detected segment(s)…", len(segments))
        combined = run_segments_pipeline(
            pdf=args.pdf,
            coords_dir=args.coords_dir,
            segments=segments,
            crops_root=args.crops_dir,
            dpi=args.dpi,
            model=args.model,
            max_workers=args.max_workers,
            prompt=args.prompt,
            prompt_file=args.prompt_file,
            write_out=args.out,
            render_out=args.render_out,
            render_format=args.render_format,
            keep_crops=args.keep_crops,
            segment_workers=args.segment_workers,
            # gemini_batch_size removed
        )
        log.info("Extraction complete.")

        # attach diagnostics
        combined["detected_forms"] = detected
        combined["missing_coord_mappings"] = sorted(set(missing_coords))
        if args.missing_coords_out and combined["missing_coord_mappings"]:
            _write_missing_coords(args.missing_coords_out, combined["missing_coord_mappings"])

        # optional splitting
        if args.split_out:
            log.info("Splitting PDF into segments → %s", args.split_out)
            spans = [
                SplitSpec(
                    label=d["form_number"].replace(" ", "_"),
                    start_page=s.start_page,
                    end_page=s.end_page,
                )
                for d, s in zip(detected, segments)
            ]
            written = split_pdf_by_spans(
                pdf_path=args.pdf,
                spans=spans,
                out_dir=args.split_out,
                pattern=args.split_pattern,
                overwrite=bool(args.split_overwrite),
            )
            combined["split_outputs"] = written
            log.info("Split outputs: %d file(s).", len(written))

        # print only the combined rendered output to STDOUT
        print(combined.get("rendered_only_extractions", ""))
        log.info("Done.")
        return

    # --- SEGMENTS MODE ---
    if args.segments:
        seg_list = json.loads(Path(args.segments).read_text(encoding="utf-8"))
        log.info("Segments mode: %d segment(s) from %s", len(seg_list), args.segments)
        segments: List[SegmentSpec] = []
        for s in seg_list:
            segments.append(SegmentSpec(
                form_number=s["form_number"],
                start_page=int(s["start_page"]),
                end_page=int(s["end_page"]),
                coords_json=Path(s["coords_json"]).expanduser().resolve()
                    if s.get("coords_json") else None,
                edition=s.get("edition") or args.default_edition,
                jurisdiction=s.get("jurisdiction") or args.default_jurisdiction,
            ))

        # Normalize pages against actual PDF length
        segments, page_fixes = _validate_and_normalize_segments(args.pdf, segments)
        for fx in page_fixes:
            log.warning('Page fix for %s: %s → %s (%s)',
                        fx.form_number, fx.original, fx.fixed, fx.reason)

        # Identification-style logs for segments
        seg_detect_like: List[DetectedForm] = []
        for s in segments:
            seg_detect_like.append({
                "form_number": s.form_number,
                "form_title": "",
                "edition": s.edition or "",
                "start_page": s.start_page,
                "end_page": s.end_page,
                "confidence": 1.0,
                "raw_markings": [],
                "potential_coord_mapping": str(s.coords_json) if s.coords_json else "",
            })
        log_acord_identities(seg_detect_like, source="segments", pdf=args.pdf, coords_dir=args.coords_dir)
        if args.detected_log:
            _write_detected_log(args.detected_log, seg_detect_like)

        combined = run_segments_pipeline(
            pdf=args.pdf,
            coords_dir=args.coords_dir,
            segments=segments,
            crops_root=args.crops_dir,
            dpi=args.dpi,
            model=args.model,
            max_workers=args.max_workers,
            prompt=args.prompt,
            prompt_file=args.prompt_file,
            write_out=args.out,
            render_out=args.render_out,
            render_format=args.render_format,
            keep_crops=args.keep_crops,
            segment_workers=args.segment_workers,
            # gemini_batch_size removed
        )
        log.info("Segments extraction complete.")

        # optional splitting (segments.json could drive splitting)
        if args.split_out:
            log.info("Splitting PDF according to segments.json → %s", args.split_out)
            spans = [
                SplitSpec(
                    label=s["form_number"].replace(" ", "_"),
                    start_page=int(s["start_page"]),
                    end_page=int(s["end_page"]),
                )
                for s in seg_list
            ]
            written = split_pdf_by_spans(
                pdf_path=args.pdf,
                spans=spans,
                out_dir=args.split_out,
                pattern=args.split_pattern,
                overwrite=bool(args.split_overwrite),
            )
            log.info("Split outputs: %d file(s).", len(written))

        print(combined.get("rendered_only_extractions", ""))
        log.info("Done.")
        return

    # --- SINGLE-RUN MODE ---
    log.info("Single-run mode starting…")
    payload = run_pipeline(
        pdf=args.pdf,
        coords_dir=args.coords_dir,
        crops_dir=args.crops_dir,
        coords_json=args.coords_json,
        dpi=args.dpi,
        prefix=args.prefix,
        model=args.model,
        max_workers=args.max_workers,
        prompt=args.prompt,
        prompt_file=args.prompt_file,
        write_out=args.out,
        render_out=args.render_out,
        render_format=args.render_format,
        keep_crops=args.keep_crops,
        # gemini_batch_size removed
    )
    log.info("Single-run extraction complete.")
    print(payload["rendered_only_extractions"])
    log.info("Done.")


if __name__ == "__main__":
    _main_cli()
