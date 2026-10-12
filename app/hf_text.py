"""Hugging Face hosted text generation (Gemini alternative).

Provides the same two text stages as the Gemini modules — a 5-panel
outline and the detailed comic expansion — through Hugging Face
Inference Providers (`InferenceClient.chat_completion`, provider
`auto`). No local model execution, no downloads.

Stage semantics (prompts, schemas, validation rules) are reused from the
Gemini modules so both providers produce interchangeable results:
`generate_outline()` returns a validated `ComicOutline` and
`generate_detailed_comic()` returns a validated `DetailedComic`, using
the exact same call signatures. Only the transport here is
Hugging Face-specific. Never logs secrets or auth headers.
"""

import json
from typing import Any

from pydantic import BaseModel, ValidationError

from app.config import (
    get_hf_text_model,
    get_hf_token,
    get_max_attempts,
    get_timeout_seconds,
)
from app.gemini_flash import (
    SYSTEM_PROMPT as FLASH_SYSTEM_PROMPT,
    ComicOutline,
    build_outline_prompt,
)
from app.gemini_pro import (
    SYSTEM_PROMPT as PRO_SYSTEM_PROMPT,
    DetailedComic,
    build_detailed_prompt,
)
from app.retry import call_with_retry

PROVIDER = "auto"


class HFTextConfigError(RuntimeError):
    """Missing/invalid text-generation configuration (e.g. no HF token)."""


class HFTextValidationError(ValueError):
    """User input, outline, or model output failed validation."""


class HFTextGenerationError(RuntimeError):
    """Hugging Face text generation failed or returned unusable output."""


def _must_not_be_blank(field_name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HFTextValidationError(
            f"{field_name} must not be empty or whitespace"
        )
    return value


def _get_client(api_key: str | None = None) -> Any:
    resolved_key = (
        (api_key.strip() if api_key and api_key.strip() else None)
        or get_hf_token()
    )
    if not resolved_key:
        raise HFTextConfigError(
            "HF_TOKEN is not configured. Set it in the environment or .env."
        )
    try:
        from huggingface_hub import InferenceClient
    except ImportError as exc:
        raise HFTextGenerationError(
            "huggingface_hub is not installed. "
            "Add 'huggingface_hub' to requirements."
        ) from exc
    try:
        return InferenceClient(
            api_key=resolved_key,
            provider=PROVIDER,
            timeout=get_timeout_seconds(),
        )
    except Exception as exc:
        raise HFTextGenerationError(
            f"Failed to create Hugging Face client: {exc}"
        ) from exc


def _response_format(schema_model: type[BaseModel]) -> dict[str, Any]:
    # "strict" is deliberately omitted: strict-schema support varies by
    # provider, and the Pydantic re-validation below is the real gate.
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_model.__name__,
            "schema": schema_model.model_json_schema(),
        },
    }


def _extract_text(response: Any) -> str:
    try:
        choices = getattr(response, "choices", None)
        message = getattr(choices[0], "message", None) if choices else None
        content = getattr(message, "content", None)
    except (IndexError, AttributeError, TypeError, KeyError) as exc:
        raise HFTextGenerationError(
            f"Hugging Face returned an unexpected response shape: {exc}"
        ) from exc
    if not isinstance(content, str) or not content.strip():
        raise HFTextGenerationError("Hugging Face returned an empty response")
    return content.strip()


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as exc:
        raise HFTextValidationError(
            f"Model output was not valid JSON: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise HFTextValidationError(
            "Model output must be a JSON object at the top level"
        )
    return data


def _chat_json(
    system: str,
    user: str,
    schema_model: type[BaseModel],
    *,
    client: Any | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    resolved = client if client is not None else _get_client(api_key)
    resolved_model = (
        (model.strip() if model and model.strip() else None)
        or get_hf_text_model()
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        response = call_with_retry(
            lambda: resolved.chat_completion(
                messages=messages,
                model=resolved_model,
                response_format=_response_format(schema_model),
            ),
            max_attempts=get_max_attempts(),
        )
    except (HFTextConfigError, HFTextValidationError, HFTextGenerationError):
        raise
    except Exception as exc:
        raise HFTextGenerationError(
            f"Hugging Face text generation failed: {exc}"
        ) from exc
    return _extract_json(_extract_text(response))


def _check_outline_shape(outline: Any) -> None:
    try:
        panels = (
            outline.panels if hasattr(outline, "panels") else outline.get("panels")
        )
        title = outline.title if hasattr(outline, "title") else outline.get("title")
        _must_not_be_blank("outline.title", title)
        if not isinstance(panels, list) or len(panels) != 5:
            raise HFTextValidationError("outline must contain exactly 5 panels")
        numbers = [
            p.panel_number if hasattr(p, "panel_number") else p.get("panel_number")
            for p in panels
        ]
        if numbers != [1, 2, 3, 4, 5]:
            raise HFTextValidationError(
                "outline panels must be numbered [1, 2, 3, 4, 5] in order, "
                f"got {numbers}"
            )
    except (HFTextValidationError, ValueError, AttributeError) as exc:
        if isinstance(exc, HFTextValidationError):
            raise
        raise HFTextValidationError(f"Invalid outline: {exc}") from exc


def parse_outline_result(data: dict[str, Any]) -> ComicOutline:
    """Validate raw outline data into a ComicOutline."""
    try:
        return ComicOutline.model_validate(data)
    except ValidationError as exc:
        raise HFTextValidationError(f"Invalid comic outline: {exc}") from exc


def parse_detailed_result(data: dict[str, Any]) -> DetailedComic:
    """Validate raw detailed data into a DetailedComic."""
    try:
        return DetailedComic.model_validate(data)
    except ValidationError as exc:
        raise HFTextValidationError(f"Invalid detailed comic: {exc}") from exc


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
    """Generate a validated 5-panel outline via hosted HF text generation.

    Same signature and return type as
    :func:`app.gemini_flash.generate_outline`.

    Raises:
        HFTextValidationError: Empty user input or invalid model output.
        HFTextConfigError: No HF token available.
        HFTextGenerationError: API call failed or response was unusable.
    """
    for field_name, value in (
        ("prompt", prompt),
        ("character", character),
        ("setting", setting),
        ("tone", tone),
        ("art_style", art_style),
    ):
        _must_not_be_blank(field_name, value)

    user = build_outline_prompt(prompt, character, setting, tone, art_style)
    data = _chat_json(
        FLASH_SYSTEM_PROMPT,
        user,
        ComicOutline,
        client=client,
        model=model,
        api_key=api_key,
    )
    return parse_outline_result(data)


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
    """Expand a validated outline into detailed content via hosted HF text.

    Same signature and return type as
    :func:`app.gemini_pro.generate_detailed_comic`.

    Raises:
        HFTextValidationError: Empty user input, bad outline, or invalid
            model output.
        HFTextConfigError: No HF token available.
        HFTextGenerationError: API call failed or response was unusable.
    """
    for field_name, value in (
        ("prompt", prompt),
        ("character", character),
        ("setting", setting),
        ("tone", tone),
        ("art_style", art_style),
    ):
        _must_not_be_blank(field_name, value)

    if outline is None:
        raise HFTextValidationError("outline must not be None")
    _check_outline_shape(outline)

    user = build_detailed_prompt(
        outline, prompt, character, setting, tone, art_style
    )
    data = _chat_json(
        PRO_SYSTEM_PROMPT,
        user,
        DetailedComic,
        client=client,
        model=model,
        api_key=api_key,
    )
    return parse_detailed_result(data)
