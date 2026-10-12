"""Backend comic-generation pipeline (Phase 6).

Orchestrates the existing stage modules into one flow without rewriting
them and without embedding provider logic here:

User input
→ 5-panel outline (Gemini Flash or Hugging Face text, selectable)
→ detailed comic (Gemini Pro or Hugging Face text, selectable)
→ Hugging Face panel images (`app.image_generator`)
→ canonical `Comic` object (`app.models`) with saved `image_path` values.

No live API calls happen at import time. All AI dependencies are
patchable at `app.pipeline` module level and may also be injected per
call, so unit tests never touch Gemini or Hugging Face. No secrets are
logged or stored here.
"""

import uuid
from pathlib import Path
from typing import Any, Callable

from app import hf_text
from app.config import get_text_provider
from app.gemini_flash import generate_outline
from app.gemini_pro import generate_detailed_comic, to_comic
from app.image_generator import generate_image
from app.models import Comic

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PANELS_DIR = BASE_DIR / "static" / "panels"


class PipelineValidationError(ValueError):
    """User input or upstream result failed pipeline-level validation."""


class PipelineGenerationError(RuntimeError):
    """An AI generation stage (outline/detail/image) failed."""


class PipelinePersistenceError(RuntimeError):
    """Panel image files could not be saved."""


def _must_not_be_blank(field_name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PipelineValidationError(
            f"{field_name} must not be empty or whitespace"
        )
    return value


def _resolve_panels_dir(panels_base_dir: str | Path | None) -> Path:
    if panels_base_dir is None:
        return DEFAULT_PANELS_DIR
    return Path(panels_base_dir)


def _image_path_for(
    saved_file: Path, panels_base_dir: Path, comic_id: str, panel_number: int
) -> str:
    """Return the frontend-friendly path stored on the panel.

    Files under the project root are stored project-relative
    (e.g. ``static/panels/<id>/panel_1.png``); files outside the
    project (e.g. test temp dirs) fall back to an absolute path.
    """
    try:
        return saved_file.relative_to(BASE_DIR).as_posix()
    except ValueError:
        return saved_file.as_posix()


def generate_comic(
    prompt: str,
    character: str,
    setting: str,
    tone: str,
    art_style: str,
    *,
    outline_fn: Callable[..., Any] | None = None,
    detail_fn: Callable[..., Any] | None = None,
    image_fn: Callable[[str], Any] | None = None,
    panels_base_dir: str | Path | None = None,
) -> Comic:
    """Run the full backend pipeline and return the canonical Comic.

    Args:
        prompt: User story prompt.
        character: Main character name.
        setting: Story setting.
        tone: Story tone.
        art_style: Desired art style.
        outline_fn: Flash/HF outline generator override (tests). Defaults
            to the configured text provider
            (:func:`app.hf_text.generate_outline` or
            :func:`app.gemini_flash.generate_outline`).
        detail_fn: Pro/HF expansion override (tests). Defaults to
            the configured text provider
            (:func:`app.hf_text.generate_detailed_comic` or
            :func:`app.gemini_pro.generate_detailed_comic`).
        image_fn: Image generator override (tests). Defaults to
            :func:`app.image_generator.generate_image`.
        panels_base_dir: Panel storage root override (tests use a temp
            dir). Defaults to ``static/panels/``.

    Raises:
        PipelineValidationError: Blank user input.
        PipelineGenerationError: Outline/detail/image stage failed
            (panel failures name the panel number).
        PipelinePersistenceError: Panel image files could not be saved.
    """
    for field_name, value in (
        ("prompt", prompt),
        ("character", character),
        ("setting", setting),
        ("tone", tone),
        ("art_style", art_style),
    ):
        _must_not_be_blank(field_name, value)

    if outline_fn is None:
        if get_text_provider() == "hf":
            outline_callable = hf_text.generate_outline
        else:
            outline_callable = generate_outline
    else:
        outline_callable = outline_fn
    if detail_fn is None:
        if get_text_provider() == "hf":
            detail_callable = hf_text.generate_detailed_comic
        else:
            detail_callable = generate_detailed_comic
    else:
        detail_callable = detail_fn
    image_callable = image_fn or generate_image
    base_dir = _resolve_panels_dir(panels_base_dir)

    try:
        outline = outline_callable(
            prompt=prompt,
            character=character,
            setting=setting,
            tone=tone,
            art_style=art_style,
        )
    except Exception as exc:
        raise PipelineGenerationError(
            f"Outline generation failed: {exc}"
        ) from exc

    try:
        detailed = detail_callable(
            outline=outline,
            prompt=prompt,
            character=character,
            setting=setting,
            tone=tone,
            art_style=art_style,
        )
    except Exception as exc:
        raise PipelineGenerationError(
            f"Detailed generation failed: {exc}"
        ) from exc

    try:
        comic = to_comic(
            detailed,
            prompt=prompt,
            character=character,
            setting=setting,
            tone=tone,
            art_style=art_style,
        )
    except Exception as exc:
        raise PipelineGenerationError(
            f"Failed to build canonical Comic: {exc}"
        ) from exc

    comic_id = uuid.uuid4().hex
    comic_dir = base_dir / comic_id
    try:
        comic_dir.mkdir(parents=True, exist_ok=False)
    except Exception as exc:
        raise PipelinePersistenceError(
            f"Could not create panel directory for comic {comic_id}: {exc}"
        ) from exc

    for panel in comic.panels:
        number = panel.panel_number
        try:
            image = image_callable(panel.image_prompt)
        except Exception as exc:
            raise PipelineGenerationError(
                f"Image generation failed for panel {number}: {exc}"
            ) from exc
        saved_file = comic_dir / f"panel_{number}.png"
        try:
            image.save(saved_file, format="PNG")
        except Exception as exc:
            raise PipelinePersistenceError(
                f"Could not save image for panel {number} "
                f"of comic {comic_id}: {exc}"
            ) from exc
        panel.image_path = _image_path_for(
            saved_file, base_dir, comic_id, number
        )

    return comic
