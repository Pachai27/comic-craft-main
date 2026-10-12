"""Tests for Gemini Pro detailed expansion (no real API calls).

Run with the project .venv:
    python -m unittest discover -s tests -v
"""

import json
import unittest
from unittest.mock import patch

from app.gemini_flash import ComicOutline
from app.gemini_pro import (
    DetailedComic,
    DetailedValidationError,
    GeminiAPIError,
    GeminiConfigError,
    build_detailed_prompt,
    generate_detailed_comic,
    parse_detailed,
    to_comic,
)
from app.models import Comic


def make_detailed_panel(panel_number: int, **overrides) -> dict:
    data: dict = {
        "panel_number": panel_number,
        "title": f"Detailed panel {panel_number}",
        "scene_description": f"Expanded scene for panel {panel_number}",
        "narration": f"Narration for panel {panel_number}",
        "dialogue": f"Dialogue for panel {panel_number}",
        "image_prompt": f"Image prompt for panel {panel_number}, cartoon style",
    }
    data.update(overrides)
    return data


def make_detailed_data(**overrides) -> dict:
    data: dict = {
        "title": "Test Detailed Comic",
        "panels": [make_detailed_panel(i) for i in (1, 2, 3, 4, 5)],
    }
    data.update(overrides)
    return data


def make_outline() -> ComicOutline:
    return ComicOutline.model_validate(
        {
            "title": "Test Outline",
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


class FakeResponse:
    def __init__(self, *, text: str | None = None, parsed=None) -> None:
        self.text = text
        self.parsed = parsed


class FakeModels:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self._response = response
        self._error = error
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response


class FakeClient:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.models = FakeModels(response=response, error=error)


class TestDetailedParsing(unittest.TestCase):
    def test_valid_structured_pro_output_succeeds(self) -> None:
        detailed = parse_detailed(make_detailed_data())
        self.assertIsInstance(detailed, DetailedComic)
        self.assertEqual(len(detailed.panels), 5)
        reparsed = parse_detailed(json.dumps(make_detailed_data()))
        self.assertEqual(reparsed.title, "Test Detailed Comic")

    def test_four_panels_fail(self) -> None:
        with self.assertRaises(DetailedValidationError):
            parse_detailed(
                make_detailed_data(
                    panels=[make_detailed_panel(i) for i in (1, 2, 3, 4)]
                )
            )

    def test_six_panels_fail(self) -> None:
        with self.assertRaises(DetailedValidationError):
            parse_detailed(
                make_detailed_data(
                    panels=[make_detailed_panel(i) for i in (1, 2, 3, 4, 5, 6)]
                )
            )

    def test_duplicate_panel_numbers_fail(self) -> None:
        with self.assertRaises(DetailedValidationError):
            parse_detailed(
                make_detailed_data(
                    panels=[make_detailed_panel(n) for n in (1, 2, 2, 4, 5)]
                )
            )

    def test_out_of_order_or_missing_numbers_fail(self) -> None:
        with self.assertRaises(DetailedValidationError):
            parse_detailed(
                make_detailed_data(
                    panels=[make_detailed_panel(n) for n in (5, 4, 3, 2, 1)]
                )
            )
        with self.assertRaises(DetailedValidationError):
            parse_detailed(
                make_detailed_data(
                    panels=[make_detailed_panel(n) for n in (1, 2, 3, 4, 6)]
                )
            )

    def test_blank_required_fields_fail(self) -> None:
        with self.assertRaises(DetailedValidationError):
            parse_detailed(make_detailed_data(title="   "))
        for field in (
            "title",
            "scene_description",
            "narration",
            "dialogue",
            "image_prompt",
        ):
            with self.subTest(panel_field=field):
                bad = make_detailed_data()
                bad["panels"][0][field] = "  "
                with self.assertRaises(DetailedValidationError):
                    parse_detailed(bad)


class TestGenerateDetailedComic(unittest.TestCase):
    def test_api_configuration_error_is_handled(self) -> None:
        with patch(
            "app.gemini_pro.get_gemini_api_key", return_value=None
        ):
            with self.assertRaises(GeminiConfigError):
                generate_detailed_comic(
                    make_outline(), **valid_inputs(), client=None
                )
        failing = FakeClient(error=RuntimeError("boom"))
        with self.assertRaises(GeminiAPIError):
            generate_detailed_comic(
                make_outline(), **valid_inputs(), client=failing
            )

    def test_empty_model_response_is_handled(self) -> None:
        client = FakeClient(response=FakeResponse(text="   ", parsed=None))
        with self.assertRaises(GeminiAPIError):
            generate_detailed_comic(
                make_outline(), **valid_inputs(), client=client
            )

    def test_mocked_success_requests_structured_output(self) -> None:
        client = FakeClient(
            response=FakeResponse(parsed=make_detailed_data())
        )
        detailed = generate_detailed_comic(
            make_outline(), **valid_inputs(), client=client
        )
        self.assertIsInstance(detailed, DetailedComic)
        config = client.models.calls[0].get("config", {})
        self.assertEqual(config.get("response_mime_type"), "application/json")
        self.assertIn("response_schema", config)

    def test_conversion_into_canonical_comic_succeeds(self) -> None:
        detailed = parse_detailed(make_detailed_data())
        comic = to_comic(detailed, **valid_inputs())
        self.assertIsInstance(comic, Comic)
        self.assertEqual(comic.title, "Test Detailed Comic")
        self.assertEqual(comic.prompt, valid_inputs()["prompt"])
        self.assertEqual(comic.character, "Nova")
        for panel, source in zip(comic.panels, detailed.panels):
            self.assertEqual(panel.panel_number, source.panel_number)
            self.assertEqual(panel.narration, source.narration)
            self.assertEqual(panel.dialogue, source.dialogue)
            self.assertEqual(panel.image_prompt, source.image_prompt)
            self.assertIsNone(panel.image_path)

    def test_resulting_canonical_comic_has_five_panels(self) -> None:
        client = FakeClient(
            response=FakeResponse(
                text=json.dumps(make_detailed_data()), parsed=None
            )
        )
        detailed = generate_detailed_comic(
            make_outline(), **valid_inputs(), client=client
        )
        comic = to_comic(detailed, **valid_inputs())
        self.assertEqual(len(comic.panels), 5)
        self.assertEqual(
            [p.panel_number for p in comic.panels], [1, 2, 3, 4, 5]
        )

    def test_prompt_preserves_outline_without_images(self) -> None:
        text = build_detailed_prompt(make_outline(), **valid_inputs())
        self.assertIn("Nova", text)
        self.assertIn("Panel 1:", text)
        self.assertIn("Panel 5:", text)
        self.assertNotIn("image file", text.lower())


if __name__ == "__main__":
    unittest.main()
