"""HTTP routes for ComicCraft. Keep handlers thin."""

import uuid

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from app.exporters import (
    ExportFileError,
    ExportGenerationError,
    ExportValidationError,
    export_comic_to_pdf,
)
from app.models import Comic
from app.pipeline import (
    PipelineGenerationError,
    PipelinePersistenceError,
    PipelineValidationError,
    generate_comic,
)

FORM_FIELDS = ("prompt", "character", "setting", "tone", "art_style")

# V1 limitation: generated comics are kept in a per-process in-memory
# dictionary keyed by export ID. This avoids a database for V1 but is
# not suitable for multi-worker production deployment (each worker has
# its own memory and entries vanish on restart).
COMICS: dict[str, Comic] = {}

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    """Render the ComicCraft landing page."""
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request},
    )


@router.get("/health", response_class=JSONResponse)
def health() -> dict[str, str]:
    """Basic health check endpoint."""
    return {"status": "ok"}


def _form_text(form: object, name: str) -> str:
    value = form.get(name) if hasattr(form, "get") else None
    return value if isinstance(value, str) else ""


@router.post("/generate", response_class=HTMLResponse)
async def generate(request: Request) -> HTMLResponse:
    """Handle the story form: run the pipeline, render the preview."""
    templates = request.app.state.templates
    form = await request.form()
    inputs = {name: _form_text(form, name) for name in FORM_FIELDS}
    try:
        comic = generate_comic(**inputs)
    except PipelineValidationError as exc:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"request": request, "error": str(exc), **inputs},
            status_code=400,
        )
    except (PipelineGenerationError, PipelinePersistenceError):
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
                "error": "Comic generation failed. Please try again later.",
                **inputs,
            },
            status_code=500,
        )
    return templates.TemplateResponse(
        request=request,
        name="comic_preview.html",
        context={
            "request": request,
            "comic": comic,
            "comic_id": _store_comic(comic),
        },
    )


def _store_comic(comic: Comic) -> str:
    """Store a generated Comic for later export; return its export ID."""
    comic_id = uuid.uuid4().hex
    COMICS[comic_id] = comic
    return comic_id


@router.post("/export", response_model=None)
async def export(request: Request):
    """Export a previously generated Comic to PDF (no regeneration)."""
    templates = request.app.state.templates
    form = await request.form()
    comic_id = _form_text(form, "comic_id")
    comic = COMICS.get(comic_id)
    if comic is None:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
                "error": "Comic not found. Please generate a comic first.",
            },
            status_code=404,
        )
    try:
        pdf_path = export_comic_to_pdf(comic)
    except (ExportValidationError, ExportGenerationError, ExportFileError):
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
                "error": "PDF export failed. Please try again later.",
            },
            status_code=500,
        )
    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=pdf_path.name,
    )
