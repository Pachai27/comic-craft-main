"""Hosted Hugging Face image generation (Phase 5).

Single-responsibility module: turns an image prompt into a PIL image via
Hugging Face Inference Providers. No local diffusion (no CUDA GPU on this
machine), no file persistence here — saving files belongs to a later
pipeline layer. Caller code must not depend on provider-specific details.
"""

from typing import Any

from PIL import Image

from app.config import (
    get_hf_image_model,
    get_hf_token,
    get_max_attempts,
    get_timeout_seconds,
)
from app.retry import call_with_retry

PROVIDER = "auto"


class ImageConfigError(RuntimeError):
    """Missing/invalid image-generation configuration (e.g. no HF token)."""


class ImageValidationError(ValueError):
    """Image prompt failed validation."""


class ImageGenerationError(RuntimeError):
    """Hugging Face image generation failed or returned unusable output."""


def _must_not_be_blank(field_name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ImageValidationError(f"{field_name} must not be empty or whitespace")
    return value


def generate_image(
    prompt: str,
    *,
    client: Any | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> Image.Image:
    """Generate a PIL image from a text prompt via hosted HF inference.

    Args:
        prompt: Text-to-image prompt (e.g. a panel `image_prompt`).
        client: Optional pre-configured InferenceClient (used by tests to
            avoid real API calls). When None, a client is created with
            `provider="auto"` from the HF token and the configured
            request timeout; injected clients manage their own timeouts.
        model: Optional model ID override. Defaults to HF_IMAGE_MODEL
            or the current text-to-image default.
        api_key: Optional HF token override. Defaults to HF_TOKEN.

    Raises:
        ImageValidationError: Empty prompt.
        ImageConfigError: No HF token available.
        ImageGenerationError: API call failed or response was unusable.
    """
    _must_not_be_blank("prompt", prompt)

    resolved_model = (model.strip() if model and model.strip() else None) or get_hf_image_model()

    if client is None:
        resolved_key = (api_key.strip() if api_key and api_key.strip() else None) or get_hf_token()
        if not resolved_key:
            raise ImageConfigError(
                "HF_TOKEN is not configured. Set it in the environment or .env."
            )
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise ImageGenerationError(
                "huggingface_hub is not installed. Add 'huggingface_hub' to requirements."
            ) from exc
        try:
            client = InferenceClient(
                api_key=resolved_key,
                provider=PROVIDER,
                timeout=get_timeout_seconds(),
            )
        except Exception as exc:
            raise ImageGenerationError(
                f"Failed to create Hugging Face client: {exc}"
            ) from exc

    try:
        image = call_with_retry(
            lambda: client.text_to_image(prompt, model=resolved_model),
            max_attempts=get_max_attempts(),
        )
    except (ImageConfigError, ImageValidationError, ImageGenerationError):
        raise
    except Exception as exc:
        raise ImageGenerationError(
            f"Hugging Face image generation failed: {exc}"
        ) from exc

    if image is None:
        raise ImageGenerationError("Hugging Face returned no image")
    return image
