"""Canonical Pydantic data models for ComicCraft.

`Comic` is the single source of truth for preview, export, and
future AI pipeline stages. Keep this module independent from
FastAPI routes and provider-specific (Gemini/Hugging Face) code.
"""

from pydantic import BaseModel, Field, field_validator, model_validator


def _must_not_be_blank(field_name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty or whitespace")
    return value


class Panel(BaseModel):
    """A single comic panel."""

    panel_number: int = Field(..., gt=0, description="1-based panel index, must be positive")
    title: str
    scene_description: str
    narration: str
    dialogue: str
    image_prompt: str
    image_path: str | None = None

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

    @field_validator("image_path")
    @classmethod
    def _validate_image_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _must_not_be_blank("image_path", value)


class Comic(BaseModel):
    """Canonical 5-panel comic object."""

    title: str
    prompt: str
    character: str
    setting: str
    tone: str
    art_style: str
    panels: list[Panel] = Field(..., min_length=5, max_length=5)

    @field_validator(
        "title",
        "prompt",
        "character",
        "setting",
        "tone",
        "art_style",
    )
    @classmethod
    def _validate_required_text(cls, value: str, info) -> str:
        return _must_not_be_blank(info.field_name, value)

    @model_validator(mode="after")
    def _validate_panels_sequence(self) -> "Comic":
        numbers = [p.panel_number for p in self.panels]
        if numbers != [1, 2, 3, 4, 5]:
            raise ValueError(
                "panels must contain panel numbers [1, 2, 3, 4, 5] in order, "
                f"got {numbers}"
            )
        return self
