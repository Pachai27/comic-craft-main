"""Environment-based configuration for ComicCraft.

Reads secrets and model IDs from environment variables only.
Never hard-code or log secret values.
"""

import os
from pathlib import Path

# Currently documented stable Flash model (GA). Overridable via
# GEMINI_FLASH_MODEL so the provider/model can change without code edits.
DEFAULT_FLASH_MODEL = "gemini-3.8-flash"

# Current documented Pro model for deep-reasoning expansion work.
# Overridable via GEMINI_PRO_MODEL so it can change without code edits.
DEFAULT_PRO_MODEL = "gemini-3.1-pro-preview"

# Default text-to-image model for hosted Hugging Face Inference Providers.
# Used throughout the official HF docs with provider="auto"; fast distilled
# model suitable for iterative comic-panel generation. Overridable via
# HF_IMAGE_MODEL so the provider/model can change without code edits.
DEFAULT_HF_IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"

# Default text-generation model for hosted Hugging Face Inference Providers.
# Featured in the official Inference Providers quick-start and recommended
# LLM list, with strong tool-calling/structured-output support. Served by
# multiple providers with automatic failover via provider="auto".
# Overridable via HF_TEXT_MODEL so the model can change without code edits.
DEFAULT_HF_TEXT_MODEL = "openai/gpt-oss-120b"

# Which hosted text provider the pipeline uses for outline + detailed
# stages: "hf" (Hugging Face Inference Providers) or "gemini".
# Defaults to "hf" so generation works without Gemini quota.
# Overridable via COMIC_TEXT_PROVIDER.
DEFAULT_TEXT_PROVIDER = "hf"

# Maximum total attempts for a single provider API call (initial try
# plus bounded immediate retries). Conservative default; the retry helper
# caps this at 2 so web requests never wait long. Overridable via
# AI_MAX_ATTEMPTS.
DEFAULT_MAX_ATTEMPTS = 2

# Per-request timeout for external AI provider calls, in seconds.
# Bounds each raw SDK attempt so a hung provider cannot block a web
# request indefinitely. Overridable via AI_TIMEOUT_SECONDS.
DEFAULT_TIMEOUT_SECONDS = 60


def _load_dotenv() -> None:
    """Load a local `.env` file into os.environ without overriding existing vars.

    Minimal loader to avoid adding an extra dependency. Never prints values.
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        # A missing/unreadable .env must not crash the app; env vars may
        # still be provided by the real environment.
        return


_load_dotenv()


def get_gemini_api_key() -> str | None:
    """Return the Gemini API key, or None if not configured."""
    value = os.getenv("GEMINI_API_KEY")
    if value is None or not value.strip():
        return None
    return value.strip()


def get_flash_model() -> str:
    """Return the configured Flash model ID, falling back to the default."""
    value = os.getenv("GEMINI_FLASH_MODEL")
    if value is None or not value.strip():
        return DEFAULT_FLASH_MODEL
    return value.strip()


def get_pro_model() -> str:
    """Return the configured Pro model ID, falling back to the default."""
    value = os.getenv("GEMINI_PRO_MODEL")
    if value is None or not value.strip():
        return DEFAULT_PRO_MODEL
    return value.strip()


def get_hf_token() -> str | None:
    """Return the Hugging Face token, or None if not configured."""
    value = os.getenv("HF_TOKEN")
    if value is None or not value.strip():
        return None
    return value.strip()


def get_hf_image_model() -> str:
    """Return the configured HF text-to-image model ID, or the default."""
    value = os.getenv("HF_IMAGE_MODEL")
    if value is None or not value.strip():
        return DEFAULT_HF_IMAGE_MODEL
    return value.strip()


def get_hf_text_model() -> str:
    """Return the configured HF text-generation model ID, or the default."""
    value = os.getenv("HF_TEXT_MODEL")
    if value is None or not value.strip():
        return DEFAULT_HF_TEXT_MODEL
    return value.strip()


def get_text_provider() -> str:
    """Return the hosted text provider ("hf" or "gemini")."""
    value = os.getenv("COMIC_TEXT_PROVIDER")
    if value is None or not value.strip():
        return DEFAULT_TEXT_PROVIDER
    normalized = value.strip().lower()
    if normalized not in ("hf", "gemini"):
        return DEFAULT_TEXT_PROVIDER
    return normalized


def get_max_attempts() -> int:
    """Return total provider attempts allowed (clamped to 1..2)."""
    raw = os.getenv("AI_MAX_ATTEMPTS")
    if raw is None or not raw.strip():
        return DEFAULT_MAX_ATTEMPTS
    try:
        value = int(raw.strip())
    except ValueError:
        return DEFAULT_MAX_ATTEMPTS
    return max(1, min(value, 2))


def get_timeout_seconds() -> float:
    """Return per-request provider timeout in seconds (clamped 1..300)."""
    raw = os.getenv("AI_TIMEOUT_SECONDS")
    if raw is None or not raw.strip():
        return float(DEFAULT_TIMEOUT_SECONDS)
    try:
        value = float(raw.strip())
    except ValueError:
        return float(DEFAULT_TIMEOUT_SECONDS)
    if value != value or value in (float("inf"), float("-inf")):
        return float(DEFAULT_TIMEOUT_SECONDS)
    return max(1.0, min(value, 300.0))
