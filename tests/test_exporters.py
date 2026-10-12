"""Tests for PDF export (no AI calls, temp dirs only).

Run with the project .venv:
    python -m unittest discover -s tests -v
"""

import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fpdf import FPDF
from PIL import Image

from app.exporters import (
    ExportFileError,
    ExportValidationError,
    export_comic_to_pdf,
)
from app.models import Comic

COLORS = ["red", "green", "blue", "yellow", "purple"]


def make_images(image_dir: Path) -> list[Path]:
    image_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, color in enumerate(COLORS, start=1):
        path = image_dir / f"panel_{i}.png"
        Image.new("RGB", (200, 150), color=color).save(path, format="PNG")
        paths.append(path)
    return paths


def make_comic(image_paths: list[Path], **overrides) -> Comic:
    data: dict = {
        "title": "Export Test Comic",
        "prompt": "A brave explorer finds a floating city",
        "character": "Nova",
        "setting": "Floating city",
        "tone": "Adventurous",
        "art_style": "Cartoon",
        "panels": [
            {
                "panel_number": i,
                "title": f"Panel {i} Title",
                "scene_description": f"Scene {i}",
                "narration": f"Narration {i}",
                "dialogue": f"Dialogue {i}",
                "image_prompt": f"Image prompt {i}",
                "image_path": path.as_posix(),
            }
            for i, path in zip((1, 2, 3, 4, 5), image_paths)
        ],
    }
    data.update(overrides)
    return Comic.model_validate(data)


class TestExporters(unittest.TestCase):
    def test_valid_comic_produces_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            images = make_images(Path(tmp) / "panels")
            comic = make_comic(images)
            pdf_path = export_comic_to_pdf(comic, output_dir=Path(tmp) / "out")
            self.assertTrue(pdf_path.is_file())
            self.assertEqual(pdf_path.suffix, ".pdf")
            self.assertGreater(pdf_path.stat().st_size, 0)

    def test_pdf_path_inside_requested_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "exports"
            images = make_images(Path(tmp) / "panels")
            pdf_path = export_comic_to_pdf(make_comic(images), output_dir=out_dir)
            self.assertEqual(pdf_path.parent, out_dir)

    def test_filename_is_timestamped_and_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            images = make_images(Path(tmp) / "panels")
            comic = make_comic(images, title="My Comic: A/B Story!")
            pdf_path = export_comic_to_pdf(comic, output_dir=tmp)
            self.assertRegex(
                pdf_path.name,
                r"^comic_[\w.\-]+_\d{8}-\d{6}_[0-9a-f]{8}\.pdf$",
            )
            self.assertNotIn(" ", pdf_path.name)
            self.assertNotIn("/", pdf_path.name)

    def test_all_five_panels_are_processed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            images = make_images(Path(tmp) / "panels")
            comic = make_comic(images)
            calls: list = []
            original = FPDF.image

            def spy(self, *args, **kwargs):
                calls.append((args, kwargs))
                return original(self, *args, **kwargs)

            with patch.object(FPDF, "image", spy):
                export_comic_to_pdf(comic, output_dir=tmp)
            self.assertEqual(len(calls), 5)
            used = " ".join(
                str(a) + str(k) for a, k in calls
            )
            for i in (1, 2, 3, 4, 5):
                self.assertIn(f"panel_{i}.png", used)

    def test_images_are_embedded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            images = make_images(Path(tmp) / "panels")
            pdf_path = export_comic_to_pdf(
                make_comic(images), output_dir=tmp
            )
            raw = pdf_path.read_bytes()
            self.assertEqual(raw.count(b"/Subtype /Image"), 5)

    def test_missing_image_raises_export_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            images = make_images(Path(tmp) / "panels")
            comic = make_comic(images)
            comic.panels[2].image_path = str(Path(tmp) / "nope.png")
            with self.assertRaises(ExportFileError) as ctx:
                export_comic_to_pdf(comic, output_dir=tmp)
            self.assertIn("3", str(ctx.exception))

    def test_invalid_comic_data_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ExportValidationError):
                export_comic_to_pdf({"title": "Broken"}, output_dir=tmp)
            with self.assertRaises(ExportValidationError):
                export_comic_to_pdf("not-a-comic", output_dir=tmp)  # type: ignore[arg-type]

    def test_long_text_does_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            images = make_images(Path(tmp) / "panels")
            comic = make_comic(images)
            comic.panels[0].narration = "Lorem ipsum dolor sit amet. " * 500
            comic.panels[0].dialogue = "So much dialogue! " * 500
            pdf_path = export_comic_to_pdf(comic, output_dir=tmp)
            self.assertTrue(pdf_path.is_file())

    def test_exporter_does_not_mutate_comic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            images = make_images(Path(tmp) / "panels")
            comic = make_comic(images)
            before = comic.model_dump()
            export_comic_to_pdf(comic, output_dir=tmp)
            self.assertEqual(comic.model_dump(), before)


if __name__ == "__main__":
    unittest.main()
