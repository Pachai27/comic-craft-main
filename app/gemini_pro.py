"""Gemini Pro detailed comic expansion (second AI stage).

Input: a validated 5-panel outline from Gemini Flash plus the original
user inputs. Output: detailed content for all five panels (narration,
dialogue, expanded scene descriptions, image prompts).

Uses the official `google-genai` SDK with structured JSON output
against a Pydantic schema. All model output is validated; failures
raise clear application-level exceptions. No image files are generated
here — `image_prompt` strings are text for a future text-to-image stage.
"""

from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from app.config import (
    get_gemini_api_key,
    get_max_attempts,
    get_pro_model,
    get_timeout_seconds,
)
from app.models import Comic
from app.retry import call_with_retry

SYSTEM_PROMPT = (
    "You are a comic-book expansion assistant. You receive a validated "
    "5-panel comic outline plus the original story inputs. Preserve the "
    "outline's story structure and panel order exactly: do not reorder, "
    "merge, split, or invent unrelated story elements. Maintain continuity "
    "across all five panels. For each panel expand the outline into: "
    "(1) a detailed visual scene description, (2) useful comic narration, "
    "(3) concise natural dialogue, and (4) a detailed text-to-image prompt "
    "describing characters, setting, composition, lighting, and the "
    "supplied art style, keeping the main character's appearance "
    "consistent across panels. Respect the supplied tone and art style. "
    "Return ONLY the requested structured data."
)


class GeminiConfigError(RuntimeError):
    """Missing/invalid Gemini configuration (e.g. no API key)."""


class DetailedValidationError(ValueError):
    """User input, outline, or Gemini Pro output failed validation."""


class GeminiAPIError(RuntimeError):
    """Gemini API call failed or returned unusable output."""


def _must_not_be_blank(field_name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty or whitespace")
    return value


class DetailedPanel(BaseModel):
    """Intermediate detailed panel (Phase 4, convertible to canonical Panel)."""

    panel_number: int
    title: str
    scene_description: str
    narration: str
    dialogue: str
    image_prompt: str

    @field_validator("panel_number")
    @classmethod
    def _validate_panel_number(cls, value: int) -> int:
        # NOTE: validated manually (not Field(gt=0)) so the generated
        # JSON Schema stays compatible with the google-genai SDK, which
        # rejects Pydantic v2's `exclusiveMinimum` form.
        if value <= 0:
            raise ValueError("panel_number must be positive")
        return value

    @field_validator(
        "title",
        "scene_description",
        "narration",
        "dialogue",
        "image_prompt",
    )
    @classmethod
    def _validate_required_text(cls, value: str, info) -> str:
        return _must_not_be_blank(info.field_name, value)


class DetailedComic(BaseModel):
    """Intermediate detailed 5-panel result returned by Gemini Pro."""

    title: str
    panels: list[DetailedPanel] = Field(..., min_length=5, max_length=5)

    @field_validator("title")
    @classmethod
    def _validate_title(cls, value: str) -> str:
        return _must_not_be_blank("title", value)

    @model_validator(mode="after")
    def _validate_panels_sequence(self) -> "DetailedComic":
        numbers = [p.panel_number for p in self.panels]
        if numbers != [1, 2, 3, 4, 5]:
            raise ValueError(
                "panels must contain panel numbers [1, 2, 3, 4, 5] in order, "
                f"got {numbers}"
            )
        return self


def _outline_panels_text(outline: Any) -> str:
    """Render outline panels as text for the Pro prompt."""
    panels = outline.panels if hasattr(outline, "panels") else outline.get("panels", [])
    lines: list[str] = []
    for panel in panels:
        if hasattr(panel, "panel_number"):
            number = panel.panel_number
            title = panel.title
            scene = panel.scene_description
        else:
            number = panel.get("panel_number")
            title = panel.get("title")
            scene = panel.get("scene_description")
        lines.append(f"Panel {number}: {title} — {scene}")
    return "\n".join(lines)


def _outline_title(outline: Any) -> str:
    if hasattr(outline, "title"):
        return outline.title  # type: ignore[no-any-return]
    return outline.get("title", "")


def build_detailed_prompt(
    outline: Any,
    prompt: str,
    character: str,
    setting: str,
    tone: str,
    art_style: str,
) -> str:
    """Build the user prompt sent to Gemini Pro."""
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Story prompt: {prompt}\n"
        f"Main character: {character}\n"
        f"Setting: {setting}\n"
        f"Tone: {tone}\n"
        f"Art style: {art_style}\n\n"
        f"Validated 5-panel outline (title: {_outline_title(outline)}):\n"
        f"{_outline_panels_text(outline)}\n"
    )


def parse_detailed(data: dict[str, Any] | str) -> DetailedComic:
    """Validate raw Pro output (dict or JSON string) into a DetailedComic."""
    try:
        if isinstance(data, str):
            return DetailedComic.model_validate_json(data)
        return DetailedComic.model_validate(data)
    except ValidationError as exc:
        raise DetailedValidationError(f"Invalid detailed comic: {exc}") from exc


def _extract_detailed(response: Any) -> DetailedComic:
    """Extract and validate a DetailedComic from an SDK response object."""
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        if isinstance(parsed, DetailedComic):
            return parsed
        if isinstance(parsed, dict):
            return parse_detailed(parsed)
        if isinstance(parsed, BaseModel):
            return parse_detailed(parsed.model_dump())
    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise GeminiAPIError("Gemini returned an empty response")
    return parse_detailed(text)


def _validate_outline(outline: Any) -> None:
    """Ensure the Flash outline looks like a valid 5-panel outline."""
    try:
        panels = outline.panels if hasattr(outline, "panels") else outline.get("panels")
        title = _outline_title(outline)
        _must_not_be_blank("outline.title", title)
        if not isinstance(panels, list) or len(panels) != 5:
            raise DetailedValidationError(
                "outline must contain exactly 5 panels"
            )
        numbers = [
            p.panel_number if hasattr(p, "panel_number") else p.get("panel_number")
            for p in panels
        ]
        if numbers != [1, 2, 3, 4, 5]:
            raise DetailedValidationError(
                "outline panels must be numbered [1, 2, 3, 4, 5] in order, "
                f"got {numbers}"
            )
    except (DetailedValidationError, ValueError):
        raise
    except Exception as exc:
        raise DetailedValidationError(f"Invalid outline: {exc}") from exc


def generate_detailed_comic(
    outline: Any,
    prompt: str,
    character: str,
    setting: str,
    tone: str,
    art_style: str,
    *,
    client: Any | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> DetailedComic:
    """Expand a validated Flash outline into detailed content via Gemini Pro.

    Args:
        outline: Validated Flash outline (ComicOutline instance or
            equivalent dict with title + 5 ordered panels).
        prompt: Original user story prompt.
        character: Main character name.
        setting: Story setting.
        tone: Story tone.
        art_style: Desired art style.
        client: Optional pre-configured genai client (used by tests to
            avoid real API calls). When None, a client is created from
            the API key.
        model: Optional model ID override. Defaults to GEMINI_PRO_MODEL
            or the current Pro default.
        api_key: Optional API key override. Defaults to GEMINI_API_KEY.

    Raises:
        DetailedValidationError: Empty user input, bad outline, or invalid
            model output.
        GeminiConfigError: No API key available.
        GeminiAPIError: API call failed or response was unusable.
    """
    for field_name, value in (
        ("prompt", prompt),
        ("character", character),
        ("setting", setting),
        ("tone", tone),
        ("art_style", art_style),
    ):
        try:
            _must_not_be_blank(field_name, value)
        except ValueError as exc:
            raise DetailedValidationError(str(exc)) from exc

    if outline is None:
        raise DetailedValidationError("outline must not be None")
    _validate_outline(outline)

    resolved_model = (model.strip() if model and model.strip() else None) or get_pro_model()

    if client is None:
        resolved_key = (api_key.strip() if api_key and api_key.strip() else None) or get_gemini_api_key()
        if not resolved_key:
            raise GeminiConfigError(
                "GEMINI_API_KEY is not configured. Set it in the environment or .env."
            )
        try:
            from google import genai as genai_sdk
        except ImportError as exc:
            raise GeminiAPIError(
                "google-genai SDK is not installed. Add 'google-genai' to requirements."
            ) from exc
        try:
            client = genai_sdk.Client(api_key=resolved_key)
        except Exception as exc:
            raise GeminiAPIError(f"Failed to create Gemini client: {exc}") from exc

    contents = build_detailed_prompt(
        outline, prompt, character, setting, tone, art_style
    )

    try:
        # HttpOptions.timeout is milliseconds; conversion keeps the
        # user-facing setting in seconds.
        timeout_ms = int(get_timeout_seconds() * 1000)
        response = call_with_retry(
            lambda: client.models.generate_content(
                model=resolved_model,
                contents=contents,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": DetailedComic,
                    "http_options": {"timeout": timeout_ms},
                },
            ),
            max_attempts=get_max_attempts(),
        )
    except (DetailedValidationError, GeminiAPIError, GeminiConfigError):
        raise
    except Exception as exc:
        raise GeminiAPIError(f"Gemini API call failed: {exc}") from exc

    try:
        return _extract_detailed(response)
    except (DetailedValidationError, GeminiAPIError):
        raise
    except ValidationError as exc:
        raise DetailedValidationError(f"Invalid detailed comic: {exc}") from exc
    except Exception as exc:
        raise GeminiAPIError(f"Failed to parse Gemini response: {exc}") from exc


def to_comic(
    detailed: DetailedComic | dict[str, Any],
    prompt: str,
    character: str,
    setting: str,
    tone: str,
    art_style: str,
) -> Comic:
    """Convert a validated Pro result into the canonical Comic object.

    Image paths are initialised to None; a later image stage fills them in.
    """
    for field_name, value in (
        ("prompt", prompt),
        ("character", character),
        ("setting", setting),
        ("tone", tone),
        ("art_style", art_style),
    ):
        try:
            _must_not_be_blank(field_name, value)
        except ValueError as exc:
            raise DetailedValidationError(str(exc)) from exc

    if isinstance(detailed, dict):
        detailed = parse_detailed(detailed)
    if not isinstance(detailed, DetailedComic):
        raise DetailedValidationError("detailed must be a DetailedComic or dict")

    try:
        return Comic.model_validate(
            {
                "title": detailed.title,
                "prompt": prompt,
                "character": character,
                "setting": setting,
                "tone": tone,
                "art_style": art_style,
                "panels": [
                    {
                        "panel_number": p.panel_number,
                        "title": p.title,
                        "scene_description": p.scene_description,
                        "narration": p.narration,
                        "dialogue": p.dialogue,
                        "image_prompt": p.image_prompt,
                        "image_path": None,
                    }
                    for p in detailed.panels
                ],
            }
        )
    except ValidationError as exc:
        raise DetailedValidationError(
            f"Failed to convert detailed result to Comic: {exc}"
        ) from exc
