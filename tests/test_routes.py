"""Tests for FastAPI routes (pipeline calls are mocked, no real AI calls).

Run with the project .venv:
    python -m unittest discover -s tests -v
"""

import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.models import Comic
from app.pipeline import PipelineGenerationError

client = TestClient(app)


def valid_form() -> dict:
    return {
        "prompt": "A brave explorer finds a floating city",
        "character": "Nova",
        "setting": "Floating city",
        "tone": "Adventurous",
        "art_style": "Cartoon",
    }


def make_comic() -> Comic:
    return Comic.model_validate(
        {
            "title": "Preview Comic",
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
                    "image_path": f"static/panels/fake-id/panel_{i}.png",
                }
                for i in (1, 2, 3, 4, 5)
            ],
        }
    )


class TestRoutes(unittest.TestCase):
    def test_index_returns_200(self) -> None:
        response = client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_index_contains_form(self) -> None:
        html = client.get("/").text
        self.assertIn("<form", html)
        for name in ("prompt", "character", "setting", "tone", "art_style"):
            self.assertIn(f'name="{name}"', html)

    def test_generate_with_mocked_pipeline_returns_200(self) -> None:
        with patch("app.routes.generate_comic", return_value=make_comic()):
            response = client.post("/generate", data=valid_form())
        self.assertEqual(response.status_code, 200)

    def test_preview_contains_five_panels(self) -> None:
        with patch("app.routes.generate_comic", return_value=make_comic()):
            html = client.post("/generate", data=valid_form()).text
        for i in (1, 2, 3, 4, 5):
            self.assertIn(f"Panel {i} Title", html)
            self.assertIn(f"Narration {i}", html)

    def test_preview_references_panel_images(self) -> None:
        with patch("app.routes.generate_comic", return_value=make_comic()):
            html = client.post("/generate", data=valid_form()).text
        for i in (1, 2, 3, 4, 5):
            self.assertIn(f"/static/panels/fake-id/panel_{i}.png", html)

    def test_invalid_form_input_shows_validation_error(self) -> None:
        bad = valid_form()
        bad["prompt"] = "   "
        response = client.post("/generate", data=bad)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Error", response.text)

    def test_pipeline_failure_shows_safe_error(self) -> None:
        with patch(
            "app.routes.generate_comic",
            side_effect=PipelineGenerationError(
                "Image generation failed for panel 3: HF_TOKEN leaked?"
            ),
        ):
            response = client.post("/generate", data=valid_form())
        self.assertEqual(response.status_code, 500)
        self.assertIn("Comic generation failed", response.text)
        for secret in ("Traceback", "HF_TOKEN", "GEMINI_API_KEY"):
            self.assertNotIn(secret, response.text)

    def test_health_returns_200(self) -> None:
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})


class TestExportRoutes(unittest.TestCase):
    def _generate_preview(self, mocked_generate) -> str:
        response = client.post("/generate", data=valid_form())
        self.assertEqual(response.status_code, 200)
        match = re.search(r'name="comic_id" value="([^"]+)"', response.text)
        self.assertIsNotNone(match)
        assert match is not None
        return match.group(1)

    def _fake_pdf(self, tmpdir: str) -> Path:
        pdf_path = Path(tmpdir) / "comic_test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake-pdf-bytes")
        return pdf_path

    def test_generation_stores_comic_for_export(self) -> None:
        from app import routes as routes_module

        with patch(
            "app.routes.generate_comic", return_value=make_comic()
        ):
            comic_id = self._generate_preview(None)
        self.assertIn(comic_id, routes_module.COMICS)
        self.assertEqual(
            routes_module.COMICS[comic_id].title, "Preview Comic"
        )

    def test_export_uses_existing_comic_and_returns_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = self._fake_pdf(tmp)
            with (
                patch(
                    "app.routes.generate_comic",
                    return_value=make_comic(),
                ) as mocked_generate,
                patch(
                    "app.routes.export_comic_to_pdf",
                    return_value=pdf_path,
                ) as mocked_export,
            ):
                comic_id = self._generate_preview(mocked_generate)
                response = client.post(
                    "/export", data={"comic_id": comic_id}
                )
            self.assertEqual(mocked_generate.call_count, 1)
            mocked_export.assert_called_once()
            exported = mocked_export.call_args[0][0]
            self.assertIsInstance(exported, Comic)
            self.assertEqual(exported.title, "Preview Comic")
            self.assertEqual(len(exported.panels), 5)
            self.assertEqual(response.status_code, 200)
            self.assertIn("application/pdf", response.headers["content-type"])
            self.assertIn(
                "attachment", response.headers["content-disposition"]
            )
            self.assertEqual(response.content, b"%PDF-1.4 fake-pdf-bytes")

    def test_export_unknown_id_handled_safely(self) -> None:
        response = client.post("/export", data={"comic_id": "nope"})
        self.assertEqual(response.status_code, 404)
        self.assertIn("Comic not found", response.text)
        for secret in ("Traceback", "HF_TOKEN", "GEMINI_API_KEY"):
            self.assertNotIn(secret, response.text)

    def test_exporter_failure_handled_safely(self) -> None:
        from app.exporters import ExportGenerationError

        with (
            patch(
                "app.routes.generate_comic", return_value=make_comic()
            ) as mocked_generate,
            patch(
                "app.routes.export_comic_to_pdf",
                side_effect=ExportGenerationError("render boom: HF_TOKEN?"),
            ),
        ):
            comic_id = self._generate_preview(mocked_generate)
            response = client.post("/export", data={"comic_id": comic_id})
        self.assertEqual(mocked_generate.call_count, 1)
        self.assertEqual(response.status_code, 500)
        self.assertIn("PDF export failed", response.text)
        for secret in ("Traceback", "HF_TOKEN", "GEMINI_API_KEY"):
            self.assertNotIn(secret, response.text)


if __name__ == "__main__":
    unittest.main()
