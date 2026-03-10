from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional

from acord_extraction import run_segments_pipeline, SegmentSpec
from main import (
    auto_detect_acord_spans,
    _resolve_coords_path,
    _validate_and_normalize_segments,
)

HERE = Path(__file__).resolve().parent
DEFAULT_COORDS_DIR = HERE / "coords"
DEFAULT_DETECT_MODEL = "gemini-2.5-flash"
DEFAULT_EXTRACTION_MODEL = "gemini-flash-lite-latest"
DEFAULT_DETECT_MAX_OUTPUT_TOKENS = 16392
DEFAULT_SEGMENT_MAX_OUTPUT_TOKENS = 10000
DEFAULT_TIMEOUT_SEC = 120.0
DEFAULT_DPI = 300
USE_CUSTOM_PROMPT = True
CUSTOM_PROMPT = "Extract the key data in a clear markdown table."


def run_default_cli(
    pdf_path: Path | str,
    *,
    coords_dir: Path | str = DEFAULT_COORDS_DIR,
    model: str = DEFAULT_EXTRACTION_MODEL,
    detect_model: Optional[str] = None,
    max_output_tokens: int = DEFAULT_DETECT_MAX_OUTPUT_TOKENS,
    segment_max_output_tokens: int = DEFAULT_SEGMENT_MAX_OUTPUT_TOKENS,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    dpi: int = DEFAULT_DPI,
    max_workers: Optional[int] = 32,
    segment_workers: Optional[int] = 32,
    prompt=CUSTOM_PROMPT if USE_CUSTOM_PROMPT else None,
) -> Dict[str, Any]:
    pdf_path = Path(pdf_path).expanduser().resolve()
    coords_dir_path = Path(coords_dir).expanduser().resolve()
    if not coords_dir_path.exists():
        raise FileNotFoundError(f"Coords directory not found: {coords_dir_path}")
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    detect_model = detect_model or DEFAULT_DETECT_MODEL

    forms = auto_detect_acord_spans(
        pdf_path=pdf_path,
        model_name=detect_model,
        max_output_tokens=max_output_tokens,
        timeout_sec=timeout_sec,
    )
    # print(forms)
    if not forms:
        return {
            "forms": [],
            "rendered_text": "",
            "result": None,
            "missing_coord_mappings": [],
            "page_fixes": [],
        }

    segments: List[SegmentSpec] = []
    missing_coords: List[str] = []
    for form in forms:
        coord_hint = form.get("potential_coord_mapping")
        coord_path = _resolve_coords_path(coords_dir_path, coord_hint)
        if coord_path is None:
            missing_coords.append(form.get("form_number", ""))
        segments.append(
            SegmentSpec(
                form_number=form.get("form_number", ""),
                start_page=int(form.get("start_page", 1)),
                end_page=int(form.get("end_page", form.get("start_page", 1))),
                coords_json=coord_path,
                edition=form.get("edition"),
                jurisdiction=None,
            )
        )

    segments, page_fixes = _validate_and_normalize_segments(pdf_path, segments)
    page_fix_payload = [pf.__dict__ for pf in page_fixes]

    with tempfile.TemporaryDirectory(prefix="acord_auto_extract_") as tmpdir:
        crops_root = Path(tmpdir) / "crops"
        crops_root.mkdir(parents=True, exist_ok=True)

        result = run_segments_pipeline(
            pdf=pdf_path,
            coords_dir=coords_dir_path,
            segments=segments,
            crops_root=crops_root,
            dpi=dpi,
            model=model,
            max_workers=max_workers,
            prompt=prompt,
            prompt_file=None,
            segment_max_output_tokens=segment_max_output_tokens,
            write_out=None,
            render_out=None,
            render_format="md",
            keep_crops=False,
            segment_workers=segment_workers,
        )

    rendered = result.get("rendered_only_extractions", "")
    extraction_details = [
        {
            "form_number": seg.get("form_number"),
            "page_span": seg.get("page_span"),
            "rendered_only_extractions": seg.get("rendered_only_extractions", ""),
            "coords_json_original": seg.get("coords_json_original"),
            "coords_json_aligned": seg.get("coords_json_aligned"),
        }
        for seg in result.get("segments", [])
    ]

    return {
        "forms": forms,
        "rendered_text": rendered,
        "result": result,
        "segments": extraction_details,
        "missing_coord_mappings": missing_coords,
        "page_fixes": page_fix_payload,
    }
