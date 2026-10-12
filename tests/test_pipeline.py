"""Tests for the backend comic pipeline (no real API calls).

Run with the project .venv:
    python -m unittest discover -s tests -v

All filesystem assertions use temporary directories; the real
`static/panels/` directory is never written to.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app.gemini_flash import ComicOutline
from app.gemini_pro import DetailedComic, DetailedValidationError
from app.models import Comic
from app.pipeline import (
    PipelineGenerationError,
    PipelinePersistenceError,
    PipelineValidationError,
    generate_comic,
)


def valid_inputs(**overrides) -> dict:
    data: dict = {
        "prompt": "A brave explorer finds a floating city",
        "character": "Nova",
        "setting": "Floating city",
        "tone": "Adventurous",
        "art_style": "Cartoon",
    }
    data.update(overrides)
    return data


def make_outline() -> ComicOutline:
    return ComicOutline.model_validate(
        {
            "title": "Outline Title",
            "panels": [
                {
                    "panel_number": i,
                    "title": f"Outline {i}",
                    "scene_description": f"Outline scene {i}",
                }
                for i in (1, 2, 3, 4, 5)
            ],
        }
    )


def make_detailed() -> DetailedComic:
    return DetailedComic.model_validate(
        {
            "title": "Detailed Title",
            "panels": [
                {
                    "panel_number": i,
                    "title": f"Panel {i}",
                    "scene_description": f"Scene {i}",
                    "narration": f"Narration {i}",
                    "dialogue": f"Dialogue {i}",
                    "image_prompt": f"Image prompt {i}, cartoon style",
                }
                for i in (1, 2, 3, 4, 5)
            ],
        }
    )


def make_image() -> Image.Image:
    return Image.new("RGB", (32, 32), color="blue")


class Fakes:
    """Recording fake stage dependencies."""

    def __init__(
        self,
        *,
        outline=None,
        detailed=None,
        outline_error: Exception | None = None,
        detail_error: Exception | None = None,
        image_error_at: int | None = None,
        image_error: Exception | None = None,
    ) -> None:
        self.outline = outline or make_outline()
        self.detailed = detailed or make_detailed()
        self.outline_error = outline_error
        self.detail_error = detail_error
        self.image_error_at = image_error_at
        self.image_error = image_error or RuntimeError("image boom")
        self.outline_calls: list[dict] = []
        self.detail_calls: list[dict] = []
        self.image_calls: list[str] = []

    def outline_fn(self, **kwargs):
        self.outline_calls.append(kwargs)
        if self.outline_error is not None:
            raise self.outline_error
        return self.outline

    def detail_fn(self, **kwargs):
        self.detail_calls.append(kwargs)
        if self.detail_error is not None:
            raise self.detail_error
        return self.detailed

    def image_fn(self, prompt: str):
        self.image_calls.append(prompt)
        if (
            self.image_error_at is not None
            and len(self.image_calls) == self.image_error_at
        ):
            raise self.image_error
        return make_image()


class BadImage:
    """Fake image whose save always fails."""

    def save(self, *args, **kwargs) -> None:
        raise OSError("disk full")


class TestPipeline(unittest.TestCase):
    def test_blank_input_rejected(self) -> None:
        for field in ("prompt", "character", "setting", "tone", "art_style"):
            with self.subTest(field=field):
                fakes = Fakes()
                with self.assertRaises(PipelineValidationError):
                    generate_comic(
                        **valid_inputs(**{field: "   "}),
                        outline_fn=fakes.outline_fn,
                        detail_fn=fakes.detail_fn,
                        image_fn=fakes.image_fn,
                    )
                self.assertEqual(fakes.outline_calls, [])
                self.assertEqual(fakes.detail_calls, [])
                self.assertEqual(fakes.image_calls, [])

    def test_successful_pipeline_returns_comic(self) -> None:
        fakes = Fakes()
        with tempfile.TemporaryDirectory() as tmp:
            comic = generate_comic(
                **valid_inputs(),
                outline_fn=fakes.outline_fn,
                detail_fn=fakes.detail_fn,
                image_fn=fakes.image_fn,
                panels_base_dir=tmp,
            )
        self.assertIsInstance(comic, Comic)
        self.assertEqual(comic.title, "Detailed Title")
        self.assertEqual(comic.character, "Nova")
        self.assertEqual(len(comic.panels), 5)

    def test_exactly_five_images_generated(self) -> None:
        fakes = Fakes()
        with tempfile.TemporaryDirectory() as tmp:
            generate_comic(
                **valid_inputs(),
                outline_fn=fakes.outline_fn,
                detail_fn=fakes.detail_fn,
                image_fn=fakes.image_fn,
                panels_base_dir=tmp,
            )
        self.assertEqual(len(fakes.image_calls), 5)
        self.assertEqual(
            fakes.image_calls,
            [f"Image prompt {i}, cartoon style" for i in (1, 2, 3, 4, 5)],
        )

    def test_five_files_created_in_comic_specific_dir(self) -> None:
        fakes = Fakes()
        with tempfile.TemporaryDirectory() as tmp:
            generate_comic(
                **valid_inputs(),
                outline_fn=fakes.outline_fn,
                detail_fn=fakes.detail_fn,
                image_fn=fakes.image_fn,
                panels_base_dir=tmp,
            )
            subdirs = [p for p in Path(tmp).iterdir() if p.is_dir()]
            self.assertEqual(len(subdirs), 1)
            files = sorted(p.name for p in subdirs[0].iterdir())
            self.assertEqual(
                files,
                [f"panel_{i}.png" for i in (1, 2, 3, 4, 5)],
            )

    def test_image_paths_match_saved_files(self) -> None:
        fakes = Fakes()
        with tempfile.TemporaryDirectory() as tmp:
            comic = generate_comic(
                **valid_inputs(),
                outline_fn=fakes.outline_fn,
                detail_fn=fakes.detail_fn,
                image_fn=fakes.image_fn,
                panels_base_dir=tmp,
            )
            parents = set()
            for panel in comic.panels:
                path = Path(panel.image_path)
                self.assertTrue(
                    path.is_file(), f"missing file for {panel.image_path}"
                )
                parents.add(path.parent)
                self.assertEqual(
                    path.name, f"panel_{panel.panel_number}.png"
                )
            self.assertEqual(len(parents), 1)

    def test_flash_failure_is_wrapped(self) -> None:
        fakes = Fakes(outline_error=RuntimeError("flash down"))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(PipelineGenerationError):
                generate_comic(
                    **valid_inputs(),
                    outline_fn=fakes.outline_fn,
                    detail_fn=fakes.detail_fn,
                    image_fn=fakes.image_fn,
                    panels_base_dir=tmp,
                )
        # Module-level functions are patchable as well (gemini path).
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.dict(os.environ, {"COMIC_TEXT_PROVIDER": "gemini"}),
                patch(
                    "app.pipeline.generate_outline",
                    side_effect=RuntimeError("flash down"),
                ),
            ):
                with self.assertRaises(PipelineGenerationError):
                    generate_comic(
                        **valid_inputs(),
                        detail_fn=Fakes().detail_fn,
                        image_fn=Fakes().image_fn,
                        panels_base_dir=tmp,
                    )

    def test_pro_failure_is_wrapped(self) -> None:
        fakes = Fakes(
            detail_error=DetailedValidationError("bad detailed output")
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(PipelineGenerationError):
                generate_comic(
                    **valid_inputs(),
                    outline_fn=fakes.outline_fn,
                    detail_fn=fakes.detail_fn,
                    image_fn=fakes.image_fn,
                    panels_base_dir=tmp,
                )

    def test_image_failure_identifies_failing_panel(self) -> None:
        fakes = Fakes(image_error_at=3)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(PipelineGenerationError) as ctx:
                generate_comic(
                    **valid_inputs(),
                    outline_fn=fakes.outline_fn,
                    detail_fn=fakes.detail_fn,
                    image_fn=fakes.image_fn,
                    panels_base_dir=tmp,
                )
        self.assertIn("panel 3", str(ctx.exception))

    def test_filesystem_failure_is_handled(self) -> None:
        fakes = Fakes()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(PipelinePersistenceError):
                generate_comic(
                    **valid_inputs(),
                    outline_fn=fakes.outline_fn,
                    detail_fn=fakes.detail_fn,
                    image_fn=lambda prompt: BadImage(),
                    panels_base_dir=tmp,
                )

    def test_panels_numbered_1_to_5_in_order(self) -> None:
        fakes = Fakes()
        with tempfile.TemporaryDirectory() as tmp:
            comic = generate_comic(
                **valid_inputs(),
                outline_fn=fakes.outline_fn,
                detail_fn=fakes.detail_fn,
                image_fn=fakes.image_fn,
                panels_base_dir=tmp,
            )
        self.assertEqual(
            [p.panel_number for p in comic.panels], [1, 2, 3, 4, 5]
        )


if __name__ == "__main__":
    unittest.main()
