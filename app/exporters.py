"""PDF export for ComicCraft (Phase 8).

Renders an already-generated canonical `Comic` object to a timestamped
PDF under ``static/exports/``. Never calls Gemini/Hugging Face, never
regenerates content, and never mutates the Comic.

Font note (V1): the project bundles no Unicode font (no `fonts/`
directory), so the exporter uses FPDF's built-in Helvetica core font and
sanitizes text to latin-1. Non-latin-1 characters render as ``?``. A
bundled Unicode TTF can replace this later without changing callers.
No network requests are made.
"""

import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from app.models import Comic

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_EXPORTS_DIR = BASE_DIR / "static" / "exports"


class ExportValidationError(ValueError):
    """Comic data failed export-time validation."""


class ExportGenerationError(RuntimeError):
    """PDF content generation failed."""


class ExportFileError(RuntimeError):
    """Export output directory/file or panel image file problem."""


def _sanitize(text: str) -> str:
    """Map text into the latin-1 range supported by FPDF core fonts."""
    return text.encode("latin-1", "replace").decode("latin-1")


def _slugify(title: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("._-")[:50]
    return slug or "untitled"


def _resolve_image(path_value: str | None, panel_number: int) -> Path:
    if not isinstance(path_value, str) or not path_value.strip():
        raise ExportFileError(f"Panel {panel_number} has no image file")
    candidate = Path(path_value.strip())
    full = candidate if candidate.is_absolute() else BASE_DIR / candidate
    if not full.is_file():
        raise ExportFileError(
            f"Panel {panel_number} image not found: {path_value}"
        )
    return full


def _validated_comic(comic: Comic | dict[str, Any]) -> Comic:
    if isinstance(comic, Comic):
        result = comic
    elif isinstance(comic, dict):
        try:
            result = Comic.model_validate(comic)
        except PydanticValidationError as exc:
            raise ExportValidationError(f"Invalid Comic data: {exc}") from exc
    else:
        raise ExportValidationError(
            f"comic must be a Comic or dict, got {type(comic).__name__}"
        )
    if len(result.panels) != 5:
        raise ExportValidationError(
            f"Comic must contain exactly 5 panels, got {len(result.panels)}"
        )
    return result


def export_comic_to_pdf(
    comic: Comic | dict[str, Any],
    output_dir: Path | str | None = None,
) -> Path:
    """Export an existing Comic to a timestamped PDF file.

    Args:
        comic: Already-generated canonical Comic (or equivalent dict).
        output_dir: Export directory override (tests use a temp dir).
            Defaults to ``static/exports/``.

    Returns:
        Path to the generated PDF file.

    Raises:
        ExportValidationError: Invalid Comic data.
        ExportFileError: Missing panel image or output file problem.
        ExportGenerationError: PDF rendering failed.
    """
    validated = _validated_comic(comic)

    target_dir = (
        DEFAULT_EXPORTS_DIR if output_dir is None else Path(output_dir)
    )
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise ExportFileError(
            f"Could not create export directory: {exc}"
        ) from exc

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"comic_{_slugify(validated.title)}_{stamp}_{uuid.uuid4().hex[:8]}.pdf"
    pdf_path = target_dir / filename

    try:
        from fpdf import FPDF
    except ImportError as exc:
        raise ExportGenerationError(
            "fpdf2 is not installed. Add 'fpdf2' to requirements."
        ) from exc

    try:
        pdf = FPDF()
        pdf.set_auto_page_break(True, margin=20)
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 22)
        pdf.multi_cell(pdf.epw, 10, _sanitize(validated.title))
        pdf.set_font("Helvetica", "", 11)
        for label, value in (
            ("Character", validated.character),
            ("Setting", validated.setting),
            ("Tone", validated.tone),
            ("Art style", validated.art_style),
        ):
            pdf.multi_cell(pdf.epw, 7, _sanitize(f"{label}: {value}"))
        pdf.ln(4)

        for panel in validated.panels:
            number = panel.panel_number
            if pdf.get_y() > 230:
                pdf.add_page()
            pdf.set_font("Helvetica", "B", 14)
            pdf.multi_cell(
                pdf.epw, 8, _sanitize(f"Panel {number}: {panel.title}")
            )
            image_file = _resolve_image(panel.image_path, number)
            try:
                if pdf.get_y() > 170:
                    pdf.add_page()
                pdf.image(str(image_file), x=30, w=150)
                pdf.ln(4)
            except ExportFileError:
                raise
            except Exception as exc:
                raise ExportFileError(
                    f"Panel {number} image could not be embedded: {exc}"
                ) from exc
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(pdf.epw, 6, _sanitize(f"Narration: {panel.narration}"))
            pdf.multi_cell(pdf.epw, 6, _sanitize(f"Dialogue: {panel.dialogue}"))
            pdf.ln(4)

        pdf.output(str(pdf_path))
    except (ExportValidationError, ExportFileError):
        raise
    except Exception as exc:
        raise ExportGenerationError(
            f"PDF generation failed: {exc}"
        ) from exc

    if not pdf_path.is_file():
        raise ExportFileError(f"PDF was not created: {pdf_path}")
    return pdf_path
