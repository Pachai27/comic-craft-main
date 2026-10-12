"""Gemini Flash 5-panel comic outline generation.

Phase 3 only: produces a structured outline (title + 5 panels with
panel_number, title, scene_description). Final narration, dialogue,
and image prompts are Phase 4 (Gemini Pro) responsibility.

Uses the official `google-genai` SDK with structured JSON output
against a Pydantic schema. All model output is validated; failures
raise clear application-level exceptions.
"""

from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from app.config import (
    get_flash_model,
    get_gemini_api_key,
    get_max_attempts,
    get_timeout_seconds,
)
from app.retry import call_with_retry

SYSTEM_PROMPT = (
    "You are a comic-book outlining assistant. Create a coherent 5-panel "
    "comic outline with clear continuity across panels. Use the supplied "
    "main character, setting, tone, and art style consistently. "
    "Return ONLY the requested structured data: a comic title and exactly "
    "5 panels numbered 1 through 5 in order, each with a short title and a "
    "1-3 sentence visual scene description. Do NOT write final narration, "
    "dialogue, or image prompts at this stage."
)


class GeminiConfigError(RuntimeError):
    """Missing/invalid Gemini configuration (e.g. no API key)."""


class OutlineValidationError(ValueError):
    """User input or Gemini outline output failed validation."""


class GeminiAPIError(RuntimeError):
    """Gemini API call failed or returned unusable output."""


def _must_not_be_blank(field_name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty or whitespace")
    return value


class OutlinePanel(BaseModel):
    """Intermediate outline panel (Phase 3 only, not the final Panel)."""

    panel_number: int
    title: str
    scene_description: str

    @field_validator("panel_number")
    @classmethod
    def _validate_panel_number(cls, value: int) -> int:
        # NOTE: validated manually (not Field(gt=0)) so the generated
        # JSON Schema stays compatible with the google-genai SDK, which
        # rejects Pydantic v2's `exclusiveMinimum: 0` form.
        if value <= 0:
            raise ValueError("panel_number must be positive")
        return value

    @field_validator("title", "scene_description")
    @classmethod
    def _validate_required_text(cls, value: str, info) -> str:
        return _must_not_be_blank(info.field_name, value)


class ComicOutline(BaseModel):
    """Intermediate 5-panel outline returned by Gemini Flash."""

    title: str
    panels: list[OutlinePanel] = Field(..., min_length=5, max_length=5)

    @field_validator("title")
    @classmethod
    def _validate_title(cls, value: str) -> str:
        return _must_not_be_blank("title", value)

    @model_validator(mode="after")
    def _validate_panels_sequence(self) -> "ComicOutline":
        numbers = [p.panel_number for p in self.panels]
        if numbers != [1, 2, 3, 4, 5]:
            raise ValueError(
                "panels must contain panel numbers [1, 2, 3, 4, 5] in order, "
                f"got {numbers}"
            )
        return self


def build_outline_prompt(
    prompt: str,
    character: str,
    setting: str,
    tone: str,
    art_style: str,
) -> str:
    """Build the user prompt sent to Gemini Flash."""
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Story prompt: {prompt}\n"
        f"Main character: {character}\n"
        f"Setting: {setting}\n"
        f"Tone: {tone}\n"
        f"Art style: {art_style}\n"
    )


def parse_outline(data: dict[str, Any] | str) -> ComicOutline:
    """Validate raw outline data (dict or JSON string) into a ComicOutline."""
    try:
        if isinstance(data, str):
            return ComicOutline.model_validate_json(data)
        return ComicOutline.model_validate(data)
    except ValidationError as exc:
        raise OutlineValidationError(f"Invalid comic outline: {exc}") from exc


def _extract_outline(response: Any) -> ComicOutline:
    """Extract and validate a ComicOutline from an SDK response object."""
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        if isinstance(parsed, ComicOutline):
            return parsed
        if isinstance(parsed, dict):
            return parse_outline(parsed)
        if isinstance(parsed, BaseModel):
            return parse_outline(parsed.model_dump())
    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise GeminiAPIError("Gemini returned an empty response")
    return parse_outline(text)


def generate_outline(
    prompt: str,
    character: str,
    setting: str,
    tone: str,
    art_style: str,
    *,
    client: Any | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> ComicOutline:
    """Generate a validated 5-panel comic outline via Gemini Flash.

    Args:
        prompt: User story prompt.
        character: Main character name.
        setting: Story setting.
        tone: Story tone.
        art_style: Desired art style.
        client: Optional pre-configured genai client (used by tests to
            avoid real API calls). When None, a client is created from
            the API key.
        model: Optional model ID override. Defaults to GEMINI_FLASH_MODEL
            or the current stable Flash default.
        api_key: Optional API key override. Defaults to GEMINI_API_KEY.

    Raises:
        OutlineValidationError: Empty user input or invalid model output.
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
        _must_not_be_blank(field_name, value)

    resolved_model = (model.strip() if model and model.strip() else None) or get_flash_model()

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

    contents = build_outline_prompt(prompt, character, setting, tone, art_style)

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
                    "response_schema": ComicOutline,
                    "http_options": {"timeout": timeout_ms},
                },
            ),
            max_attempts=get_max_attempts(),
        )
    except (OutlineValidationError, GeminiAPIError, GeminiConfigError):
        raise
    except Exception as exc:
        raise GeminiAPIError(f"Gemini API call failed: {exc}") from exc

    try:
        return _extract_outline(response)
    except (OutlineValidationError, GeminiAPIError):
        raise
    except ValidationError as exc:
        raise OutlineValidationError(f"Invalid comic outline: {exc}") from exc
    except Exception as exc:
        raise GeminiAPIError(f"Failed to parse Gemini response: {exc}") from exc
