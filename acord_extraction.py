#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
acord_extraction.py

End-to-end ACORD extraction pipeline (PDF → crops → Gemini → ordered render).

Two modes:

1) Single-run (one coords JSON)
   python acord_extraction.py \
     --pdf packet.pdf \
     --coords-dir coords/ \
     --coords-json coords/acord_25__201603__us-national.json \
     --crops-dir crops/ \
     --model gemini-2.5-flash \
     --render-out out.md \
     --out payload.json

2) Multi-segment (pre-segmented ACORD page spans; one master PDF)
   python acord_extraction.py \
     --pdf packet.pdf \
     --coords-dir coords/ \
     --crops-dir crops/ \
     --segments segments.json \
     --model gemini-2.5-flash \
     --render-out combined.md \
     --out combined.json

Where segments.json is a list like:
[
  {"form_number":"ACORD 25","start_page":1,"end_page":1,"coords_json":"coords/acord_25__201603__us-national.json"},
  {"form_number":"ACORD 28","start_page":2,"end_page":3,"coords_json":"coords/acord_28__200909__us-national.json"}
]

Dependencies (import-level):
- get_regions.crop_regions          (your existing cropper)
- gemini_extract.extract_crops      (your existing Gemini image-region extractor)

Environment:
- GENAI_API_KEY or GOOGLE_API_KEY must be set for Gemini auth (used inside gemini_extract).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# your modules
from get_regions import crop_regions
from dots_extract import extract_crops


# --------------------------- Prompt used for region extraction ---------------------------

DEFAULT_PROMPT = (
    "You will receive a cropped image of an insurance form (e.g., ACORD). "
    "Extract only the information visible within this region. "
    "There will be weirdly formatted tables. "
    "keep the paragraph and writing regions small if nothing is present. "
    "Return the result as clean Markdown. "
    "Do not include any extra commentary beyond the extracted content."
)

log = logging.getLogger("acord.pipeline")

# --------------------------------- Small utilities ---------------------------------

def _normalize_token(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

def _pdf_hint_tokens(pdf_path: Path):
    base = pdf_path.stem.lower()
    tokens = set(re.findall(r"[a-z]+|\d+", base))
    m = re.search(r"(acord)[\s_-]*(\d+)", base)
    if m:
        tokens.add(f"{m.group(1)}_{m.group(2)}")
    return {_normalize_token(t) for t in tokens if t}


# ---------------------------- Renderer (ordered output) ----------------------------

def _build_original_sequence(annotations_json_path: Path) -> List[Tuple[int, int]]:
    """
    Rebuilds the (page, idx) order used by cropping:
      - group annotations by page
      - iterate pages ascending
      - enumerate entries per page with start=1 in JSON order
    """
    data = json.loads(annotations_json_path.read_text(encoding="utf-8"))
    by_page: Dict[int, List[Dict[str, Any]]] = {}
    for a in data.get("annotations", []):
        p = int(a.get("page", 1))
        by_page.setdefault(p, []).append(a)

    seq: List[Tuple[int, int]] = []
    for page in sorted(by_page.keys()):
        anns = by_page[page]
        for idx, _ in enumerate(anns, start=1):
            seq.append((page, idx))
    return seq

def _to_plain_text(md: str) -> str:
    """
    VERY light Markdown → text cleanup (no external deps):
    - strip leading '#' headings
    - convert list items '- ' or '* ' to plain lines
    - remove codefence ticks
    - trim
    """
    lines = []
    for line in md.splitlines():
        l = line
        l = re.sub(r"^\s*#{1,6}\s*", "", l)      # remove heading marks
        l = re.sub(r"^\s*[-*]\s+", "", l)       # bullets
        l = l.replace("```", "")                # code fences
        lines.append(l)
    return "\n".join(lines).strip()

def render_only_extractions(
    payload: Dict[str, Any],
    annotations_json_path: Path,
    as_markdown: bool = True,
    separator: str = "\n\n",
) -> str:
    """
    Returns ONE big string that contains ONLY the extraction text blocks,
    ordered exactly according to the original coords JSON.
    """
    seq = _build_original_sequence(annotations_json_path)

    # Map (page,idx) -> extraction
    index: Dict[Tuple[int, int], str] = {}
    for item in payload.get("files", []):
        page = item.get("page")
        idx = item.get("idx")
        if page and idx and isinstance(item.get("extraction"), str):
            index[(int(page), int(idx))] = item["extraction"].strip()

    blocks: List[str] = []
    for key in seq:
        block = index.get(key, "")
        if block:
            blocks.append(block)

    big = separator.join(blocks).strip()
    if as_markdown:
        return big
    return _to_plain_text(big)


# ------------------------- Coords resolution + page shifting -------------------------

def resolve_coords_json(pdf_path: Path, coords_dir: Path) -> Optional[Path]:
    """
    Generic resolver: if caller doesn't specify coords_json explicitly,
    try to pick the best JSON in coords_dir by filename overlap with PDF.
    """
    if not coords_dir.exists():
        return None
    pdf_tokens = _pdf_hint_tokens(pdf_path)
    candidates = list(coords_dir.glob("*.json"))
    if not candidates:
        return None
    best = None
    best_score = -1
    for j in candidates:
        j_tokens = set(_normalize_token(t) for t in re.findall(r"[a-z0-9]+", j.stem.lower()))
        m = re.search(r"(acord)[_]*([0-9]+)", j.stem.lower())
        if m:
            j_tokens.add(f"{m.group(1)}_{m.group(2)}")
        score = len(pdf_tokens & j_tokens)
        if score > best_score:
            best = j
            best_score = score
    return best

def _norm_form_token(s: str) -> str:
    # "ACORD 25" -> "acord_25"
    m = re.search(r"(acord)\s*[-_ ]*\s*(\d+)", s or "", re.I)
    return f"acord_{m.group(2)}" if m else _normalize_token(s or "")

def resolve_coords_for_form(
    coords_dir: Path,
    form_number: str,
    edition: str | None = None,           # e.g., "2016/11"
    jurisdiction: str | None = None       # e.g., "US-National", "IL", "FL"
) -> Optional[Path]:
    """
    Edition/jurisdiction-aware resolver. Prefers filename + embedded JSON metadata matches.
    Recommended filename pattern: acord_<num>__<yyyymm>__<jurisdiction>.json
    and a metadata block { "form_number": "...", "edition": "YYYY/MM", "jurisdiction": "..." }.
    """
    if not coords_dir.exists():
        return None
    wanted_num = re.search(r"(\d+)", form_number or "")  # e.g., 80
    ed_flat = re.sub(r"[^\d]", "", edition or "")       # "2016/11" -> "201611"

    best, best_score = None, -1
    for j in coords_dir.glob("*.json"):
        name = j.stem.lower()
        score = 0
        if wanted_num and re.search(rf"\b{wanted_num.group(1)}\b", name):
            score += 2
        if ed_flat and ed_flat in name:
            score += 3
        if jurisdiction and jurisdiction.lower() in name:
            score += 2

        # peek at metadata if available
        try:
            meta = json.loads(j.read_text(encoding="utf-8")).get("metadata", {})
            if meta.get("form_number", "").replace(" ", "").lower() == (_norm_form_token(form_number).replace("_", "")):
                score += 1
            m_ed = meta.get("edition", "")
            if ed_flat and m_ed.replace("/", "") == ed_flat:
                score += 4
            m_j = meta.get("jurisdiction", "")
            if jurisdiction and m_j.lower() == jurisdiction.lower():
                score += 3
        except Exception:
            pass

        if score > best_score:
            best, best_score = j, score

    return best

def rewrite_coords_for_offset(
    orig_coords_json: Path,
    page_offset: int,
    tmp_dir: Path,
    tag: str | None = None,
) -> Path:
    """
    Rewrites the coords JSON so its 'page' fields are shifted by 'page_offset'.
    Example: coords have pages 1..N; segment starts at page P in the master PDF.
    We write a temp JSON where page i -> (i + P - 1), so crop_regions hits the right pages.
    """
    data = json.loads(orig_coords_json.read_text(encoding="utf-8"))
    for a in data.get("annotations", []):
        try:
            a["page"] = int(a.get("page", 1)) + page_offset
        except Exception:
            pass
    token = tag or f"{uuid.uuid4().hex[:8]}"
    out = tmp_dir / f"{orig_coords_json.stem}__offset_{page_offset}__{token}.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


# ----------------------------------- Single-run -----------------------------------



def run_pipeline(
    *,
    pdf: str | Path,
    coords_dir: str | Path,
    crops_dir: str | Path,
    coords_json: str | Path | None = None,
    dpi: int = 300,
    prefix: str = "",
    model: str = "gemini-flash-lite-latest",
    max_workers: int | None = None,
    prompt: str | None = None,
    prompt_file: str | Path | None = None,
    segment_max_output_tokens: int = 2048,
    write_out: str | Path | None = None,
    render_out: str | Path | None = None,
    render_format: str = "md",
    keep_crops: bool = False
) -> Dict[str, Any]:
    """
    End-to-end. Returns the payload dict. If render_out is provided, also writes a single
    MD/TXT file that contains ONLY the extractions in original order. When keep_crops is
    False (default), the cropped images created during this run will be deleted. Crops are sent to Gemini one at a time for extraction.
    """
    pdf_path = Path(pdf).expanduser().resolve()
    coords_dir_path = Path(coords_dir).expanduser().resolve()
    crops_dir_path = Path(crops_dir).expanduser().resolve()
    crops_dir_path.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    log.info("Single-run: resolving coordinates for %s", pdf_path.name)
    if coords_json is not None:
        coords_json_path = Path(coords_json).expanduser().resolve()
        log.debug("Single-run: using explicit coords JSON %s", coords_json_path)
    else:
        coords_json_path = resolve_coords_json(pdf_path, coords_dir_path)
        if not coords_json_path:
            raise FileNotFoundError(
                f"Could not resolve coords JSON from {coords_dir_path} for {pdf_path.name}. "
                f"Pass --coords-json explicitly."
            )
        log.debug("Single-run: resolved coords JSON %s", coords_json_path)

    if not coords_json_path.exists():
        raise FileNotFoundError(f"Coords JSON not found: {coords_json_path}")

    log.info("Single-run: using coords file %s", coords_json_path.name)

    # 1) Crop
    log.info("Single-run: cropping regions to %s (dpi=%d)", crops_dir_path, dpi)
    crop_result = crop_regions(
        annotations_json=coords_json_path,
        pdf_path=pdf_path,
        out_dir=crops_dir_path,
        dpi=dpi,
        prefix=prefix,
    )
    crop_count = crop_result.get("count", 0)
    log.info("Single-run: cropped %d region(s)", crop_count)

    # 2) Extract (single prompt shared across all crops)
    if prompt_file:
        prompt_text = Path(prompt_file).read_text(encoding="utf-8").strip()
    elif prompt:
        prompt_text = prompt.strip()
    else:
        prompt_text = DEFAULT_PROMPT

    log.info("Single-run: extracting crops with model %s", model)
    payload = extract_crops(
        annotations_json=coords_json_path,
        crops_dir=crops_dir_path,
        prompt=prompt_text,
        model_name=model,
        max_workers=max_workers,
        max_output_tokens=segment_max_output_tokens,
    )
    files_processed = len(payload.get("files", []))
    log.info("Single-run: extraction completed with %d item(s)", files_processed)

    # augment with context
    payload["source_pdf"] = str(pdf_path)
    payload["coords_json"] = str(coords_json_path)
    payload["crops_count"] = crop_result.get("count", 0)
    payload["crops_saved"] = crop_result.get("saved", [])

    # 3) Render only the extraction blocks in original order
    as_md = (render_format.lower() == "md")
    rendered = render_only_extractions(
        payload=payload,
        annotations_json_path=coords_json_path,
        as_markdown=as_md,
    )

    if write_out:
        out_path = Path(write_out).expanduser().resolve()
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("Single-run: wrote payload JSON to %s", out_path)

    if render_out:
        rpath = Path(render_out).expanduser().resolve()
        rpath.write_text(rendered, encoding="utf-8")
        log.info("Single-run: wrote rendered output to %s", rpath)

    payload["rendered_only_extractions"] = rendered
    payload["render_format"] = "md" if as_md else "text"

    if not keep_crops:
        deleted = 0
        failed = 0
        for fp in crop_result.get("saved", []):
            try:
                Path(fp).unlink(missing_ok=True)
                deleted += 1
            except Exception as exc:
                failed += 1
                log.warning("Single-run: failed to delete crop %s: %s", fp, exc)
        try:
            if crops_dir_path.exists() and not any(crops_dir_path.iterdir()):
                crops_dir_path.rmdir()
        except Exception as exc:
            log.debug("Single-run: unable to remove crops directory %s: %s", crops_dir_path, exc)
        log.debug("Single-run: deleted %d crop file(s); failed to delete %d", deleted, failed)
        payload["crops_deleted"] = True
        payload["crops_deleted_count"] = deleted
        payload["crops_delete_failed"] = failed
    else:
        log.debug("Single-run: keeping cropped images in %s", crops_dir_path)
        payload["crops_deleted"] = False
        payload["crops_deleted_count"] = 0
        payload["crops_delete_failed"] = 0

    return payload
# ---------------------------------- Segments mode ----------------------------------

@dataclass
class SegmentSpec:
    form_number: str              # e.g. "ACORD 25"
    start_page: int               # 1-based inclusive
    end_page: int                 # 1-based inclusive
    coords_json: Path | None = None  # explicit coords JSON for this form (optional)
    edition: str | None = None       # optional hint, e.g., "2016/11"
    jurisdiction: str | None = None  # optional hint, e.g., "US-National", "IL"





def run_segments_pipeline(
    *,
    pdf: str | Path,
    coords_dir: str | Path,
    segments: List[SegmentSpec],
    crops_root: str | Path,
    dpi: int = 300,
    model: str = "gemini-flash-lite-latest",
    max_workers: int | None = None,
    prompt: str | None = None,
    prompt_file: str | Path | None = None,
    segment_max_output_tokens: int = 2048,
    write_out: str | Path | None = None,
    render_out: str | Path | None = None,
    render_format: str = "md",
    keep_crops: bool = False,
    segment_workers: int | None = None
) -> Dict[str, Any]:
    """
    Process each segment (form_number + page span + coords) by:
      1) aligning its coordinate JSON to the master PDF,
      2) exporting crops for the specified pages,
      3) extracting the text/Markdown with Gemini,
      4) concatenating the rendered blocks in segment order,
      5) returning a combined payload.
    """
    pdf_path = Path(pdf).expanduser().resolve()
    coords_dir_path = Path(coords_dir).expanduser().resolve()
    crops_root_path = Path(crops_root).expanduser().resolve()
    crops_root_path.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not coords_dir_path.exists():
        raise FileNotFoundError(f"Coords dir not found: {coords_dir_path}")
    if not segments:
        raise ValueError("segments list is empty")

    if prompt_file:
        prompt_text = Path(prompt_file).read_text(encoding="utf-8").strip()
    elif prompt:
        prompt_text = prompt.strip()
    else:
        prompt_text = DEFAULT_PROMPT

    if segment_workers is None or segment_workers < 1:
        cpu_guess = max(1, os.cpu_count() or 1)
        segment_workers = min(len(segments), max(1, cpu_guess // 2 or 1))

    log.info(
        "Segments: preparing %d segment(s) for %s (segment workers=%d)",
        len(segments),
        pdf_path.name,
        segment_workers,
    )

    combined: Dict[str, Any] = {
        "source_pdf": str(pdf_path),
        "model": model,
        "segments": [],
        "render_format": render_format.lower(),
    }
    temp_parent = tempfile.TemporaryDirectory()
    temp_dir = Path(temp_parent.name)

    def _process_segment(idx: int, seg: SegmentSpec) -> Tuple[int, Dict[str, Any], str]:
        form = seg.form_number
        start_p, end_p = int(seg.start_page), int(seg.end_page)
        if start_p < 1 or end_p < start_p:
            raise ValueError(f"Bad page span for {form}: {start_p}-{end_p}")

        log.info("Segments: [%02d] %s pages %d-%d", idx, form, start_p, end_p)
        log.debug(
            "Segments: [%02d] resolving coords (edition=%s, jurisdiction=%s)",
            idx,
            seg.edition,
            seg.jurisdiction,
        )

        seg_coords_hint = seg.coords_json or resolve_coords_for_form(
            coords_dir_path,
            form_number=form,
            edition=seg.edition,
            jurisdiction=seg.jurisdiction,
        )
        seg_coords_path = None
        if seg_coords_hint:
            candidate = Path(seg_coords_hint)
            if not candidate.is_absolute():
                candidate = coords_dir_path / candidate
            if candidate.exists():
                seg_coords_path = candidate

        if seg_coords_path is None:
            raise FileNotFoundError(
                f"Segments: coords not found for {form} (hint={seg_coords_hint})"
            )

        log.debug("Segments: [%02d] coords source -> %s", idx, seg_coords_path)
        page_offset = start_p - 1
        aligned_coords = rewrite_coords_for_offset(
            seg_coords_path,
            page_offset,
            temp_dir,
            tag=f"seg{idx:02d}",
        )
        log.debug("Segments: [%02d] aligned coords temp file -> %s", idx, aligned_coords)
        original_coords_str = str(seg_coords_path)

        seg_crops_dir = crops_root_path / f"segment_{idx:02d}_{_norm_form_token(form)}"
        seg_crops_dir.mkdir(parents=True, exist_ok=True)

        crop_result = crop_regions(
            annotations_json=aligned_coords,
            pdf_path=pdf_path,
            out_dir=seg_crops_dir,
            dpi=dpi,
            prefix=f"{_norm_form_token(form)}_p{start_p:03d}-{end_p:03d}_",
        )
        crop_count = crop_result.get("count", 0)
        log.info("Segments: [%02d] cropped %d region(s)", idx, crop_count)

        log.info("Segments: [%02d] extracting with model %s", idx, model)
        payload = extract_crops(
            annotations_json=aligned_coords,
            crops_dir=seg_crops_dir,
            prompt=prompt_text,
            model_name=model,
            max_workers=max_workers,
            max_output_tokens=segment_max_output_tokens,
        )
        files_processed = len(payload.get("files", []))
        log.info("Segments: [%02d] extraction completed with %d item(s)", idx, files_processed)

        payload["source_pdf"] = str(pdf_path)
        payload["coords_json"] = str(aligned_coords)
        payload["coords_json_original"] = original_coords_str or None
        payload["crops_count"] = crop_result.get("count", 0)
        payload["crops_saved"] = crop_result.get("saved", [])
        payload["page_span"] = [start_p, end_p]

        as_md = (render_format.lower() == "md")
        rendered = render_only_extractions(
            payload=payload,
            annotations_json_path=aligned_coords,
            as_markdown=as_md,
        )

        if not keep_crops:
            deleted = 0
            failed = 0
            for fp in crop_result.get("saved", []):
                try:
                    Path(fp).unlink(missing_ok=True)
                    deleted += 1
                except Exception as exc:
                    failed += 1
                    log.warning("Segments: [%02d] failed to delete crop %s: %s", idx, fp, exc)
            try:
                if seg_crops_dir.exists() and not any(seg_crops_dir.iterdir()):
                    seg_crops_dir.rmdir()
            except Exception as exc:
                log.debug(
                    "Segments: [%02d] unable to remove crops directory %s: %s",
                    idx,
                    seg_crops_dir,
                    exc,
                )
            log.debug(
                "Segments: [%02d] deleted %d crop file(s); failed %d",
                idx,
                deleted,
                failed,
            )
        else:
            log.debug("Segments: [%02d] keeping cropped images in %s", idx, seg_crops_dir)

        entry = {
            "form_number": form,
            "page_span": [start_p, end_p],
            "coords_json_original": (original_coords_str or None),
            "coords_json_aligned": str(aligned_coords),
            "payload": payload,
            "rendered_only_extractions": rendered,
        }
        return idx, entry, rendered

    try:
        with ThreadPoolExecutor(max_workers=segment_workers) as pool:
            futures = {pool.submit(_process_segment, idx, seg): idx for idx, seg in enumerate(segments, start=1)}
            results: List[Tuple[int, Dict[str, Any], str]] = []
            for fut in as_completed(futures):
                results.append(fut.result())
    finally:
        temp_parent.cleanup()

    results.sort(key=lambda item: item[0])
    rendered_blocks: List[str] = []
    for _, entry, rendered in results:
        combined["segments"].append(entry)
        if rendered:
            rendered_blocks.append(rendered)

    rendered_text = ("\n\n---\n\n").join(rendered_blocks).strip()
    if render_out:
        ro = Path(render_out).expanduser().resolve()
        ro.write_text(rendered_text, encoding="utf-8")
        combined["render_out"] = str(ro)
    combined["rendered_only_extractions"] = rendered_text

    if write_out:
        wo = Path(write_out).expanduser().resolve()
        wo.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
        combined["write_out"] = str(wo)

    return combined
# -------------------------------------- CLI --------------------------------------

def _parse_args():
    ap = argparse.ArgumentParser(description="PDF → crops → Gemini → render only the extractions (text/MD).")
    # common
    ap.add_argument("--pdf", type=Path, help="Input master PDF (required in both modes)")
    ap.add_argument("--coords-dir", required=True, type=Path, help="Folder of ACORD coordinate JSONs")
    ap.add_argument("--dpi", type=int, default=300, help="DPI for crops")
    ap.add_argument("--model", type=str, default="gemini-flash-lite-latest", help="Gemini model")
    ap.add_argument("--max-workers", type=int, default=64, help="Threads for extraction (default auto)")
    ap.add_argument("--prompt", type=str, help="Single prompt to send to Gemini (text/MD output)")
    ap.add_argument("--prompt-file", type=Path, help="Load prompt from file")
    ap.add_argument("--out", type=Path, help="Write the full JSON payload to this path (optional)")
    ap.add_argument("--render-out", type=Path, help="Write only the ordered extraction text/MD here (optional)")
    ap.add_argument("--render-format", type=str, choices=["md", "text"], default="md",
                    help="Whether to render as Markdown or plain text")
    ap.add_argument("--keep-crops", action="store_true",
                    help="Keep the cropped images instead of deleting them at the end (default: delete).")

    ap.add_argument("--segment-workers", type=int, default=3,
                    help="Parallel workers for segment processing (default auto).")

    # single-run
    ap.add_argument("--coords-json", type=Path, help="(single-run) Explicit coordinates JSON (overrides resolver)")
    ap.add_argument("--crops-dir", type=Path, help="Output folder for crops (required in both modes)")

    # segments mode
    ap.add_argument("--segments", type=Path,
                    help="JSON file describing segments: "
                         "[{\"form_number\":\"ACORD 25\",\"start_page\":1,\"end_page\":1,"
                         "\"coords_json\":\"coords/acord_25__201603__us-national.json\"}, ...]")

    # optional resolution hints for segments built on-the-fly (not used if segments.json already specifies coords_json)
    ap.add_argument("--default-edition", type=str, help="Edition hint for resolver, e.g., 2016/11")
    ap.add_argument("--default-jurisdiction", type=str, help="Jurisdiction hint, e.g., US-National, IL, FL")

    return ap.parse_args()

def _main_cli():
    args = _parse_args()

    if not args.pdf:
        raise SystemExit("--pdf is required")

    if not args.crops_dir:
        raise SystemExit("--crops-dir is required")

    # SEGMENTS MODE
    if args.segments:
        seg_list = json.loads(Path(args.segments).read_text(encoding="utf-8"))
        segments: List[SegmentSpec] = []
        for s in seg_list:
            segments.append(SegmentSpec(
                form_number=s["form_number"],
                start_page=int(s["start_page"]),
                end_page=int(s["end_page"]),
                coords_json=Path(s["coords_json"]).expanduser().resolve() if s.get("coords_json") else None,
                edition=s.get("edition") or args.default_edition,
                jurisdiction=s.get("jurisdiction") or args.default_jurisdiction,
            ))

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
            segment_workers=args.segment_workers
        )
        # Print only the combined rendered output
        print(combined.get("rendered_only_extractions", ""))
        return

    # SINGLE-RUN
    payload = run_pipeline(
        pdf=args.pdf,
        coords_dir=args.coords_dir,
        crops_dir=args.crops_dir,
        coords_json=args.coords_json,
        dpi=args.dpi,
        prefix="",  # optional filename prefix for crops
        model=args.model,
        max_workers=args.max_workers,
        prompt=args.prompt,
        prompt_file=args.prompt_file,
        write_out=args.out,
        render_out=args.render_out,
        render_format=args.render_format,
        keep_crops=args.keep_crops,
    )
    print(payload["rendered_only_extractions"])


if __name__ == "__main__":
    _main_cli()



