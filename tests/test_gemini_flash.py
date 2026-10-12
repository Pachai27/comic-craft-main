"""Tests for Gemini Flash outline generation (no real API calls).

Run with the project .venv:
    python -m unittest discover -s tests -v
"""

import json
import unittest
from unittest.mock import patch

from app.gemini_flash import (
    ComicOutline,
    GeminiAPIError,
    GeminiConfigError,
    OutlineValidationError,
    build_outline_prompt,
    generate_outline,
    parse_outline,
)


def make_outline_panel(panel_number: int, **overrides) -> dict:
    data: dict = {
        "panel_number": panel_number,
        "title": f"Outline panel {panel_number}",
        "scene_description": f"Visual scene for panel {panel_number}",
    }
    data.update(overrides)
    return data


def make_outline_data(**overrides) -> dict:
    data: dict = {
        "title": "Test Outline",
        "panels": [make_outline_panel(i) for i in (1, 2, 3, 4, 5)],
    }
    data.update(overrides)
    return data


class FakeResponse:
    def __init__(
        self, *, text: str | None = None, parsed=None
    ) -> None:
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


class TestOutlineParsing(unittest.TestCase):
    def test_valid_outline_dict_succeeds(self) -> None:
        outline = parse_outline(make_outline_data())
        self.assertEqual(outline.title, "Test Outline")
        self.assertEqual(
            [p.panel_number for p in outline.panels], [1, 2, 3, 4, 5]
        )

    def test_valid_outline_json_string_succeeds(self) -> None:
        outline = parse_outline(json.dumps(make_outline_data()))
        self.assertEqual(len(outline.panels), 5)

    def test_invalid_panel_count_fails(self) -> None:
        with self.assertRaises(OutlineValidationError):
            parse_outline(
                make_outline_data(
                    panels=[make_outline_panel(i) for i in (1, 2, 3, 4)]
                )
            )
        with self.assertRaises(OutlineValidationError):
            parse_outline(
                make_outline_data(
                    panels=[make_outline_panel(i) for i in (1, 2, 3, 4, 5, 6)]
                )
            )

    def test_duplicate_panel_numbers_fail(self) -> None:
        data = make_outline_data(
            panels=[make_outline_panel(n) for n in (1, 2, 2, 4, 5)]
        )
        with self.assertRaises(OutlineValidationError):
            parse_outline(data)

    def test_out_of_order_or_missing_numbers_fail(self) -> None:
        with self.assertRaises(OutlineValidationError):
            parse_outline(
                make_outline_data(
                    panels=[make_outline_panel(n) for n in (1, 2, 3, 4, 6)]
                )
            )
        with self.assertRaises(OutlineValidationError):
            parse_outline(
                make_outline_data(
                    panels=[make_outline_panel(n) for n in (5, 4, 3, 2, 1)]
                )
            )

    def test_empty_required_fields_fail(self) -> None:
        with self.assertRaises(OutlineValidationError):
            parse_outline(make_outline_data(title="   "))
        bad = make_outline_data()
        bad["panels"][0]["title"] = "  "
        with self.assertRaises(OutlineValidationError):
            parse_outline(bad)
        bad2 = make_outline_data()
        bad2["panels"][1]["scene_description"] = ""
        with self.assertRaises(OutlineValidationError):
            parse_outline(bad2)


class TestGenerateOutline(unittest.TestCase):
    def _valid_inputs(self) -> dict:
        return {
            "prompt": "A brave explorer finds a floating city",
            "character": "Nova",
            "setting": "Floating city",
            "tone": "Adventurous",
            "art_style": "Cartoon",
        }

    def test_missing_api_key_raises_config_error(self) -> None:
        with patch(
            "app.gemini_flash.get_gemini_api_key", return_value=None
        ):
            with self.assertRaises(GeminiConfigError):
                generate_outline(**self._valid_inputs(), client=None)

    def test_empty_user_input_fails(self) -> None:
        client = FakeClient(response=FakeResponse(text="{}"))
        inputs = self._valid_inputs()
        inputs["prompt"] = "   "
        with self.assertRaises((OutlineValidationError, ValueError)):
            generate_outline(**inputs, client=client)

    def test_mocked_client_with_parsed_dict_succeeds(self) -> None:
        client = FakeClient(
            response=FakeResponse(parsed=make_outline_data())
        )
        outline = generate_outline(**self._valid_inputs(), client=client)
        self.assertIsInstance(outline, ComicOutline)
        self.assertEqual(len(outline.panels), 5)
        # Structured JSON output must have been requested.
        self.assertEqual(
            client.models.calls[0].get("config", {}).get("response_mime_type"),
            "application/json",
        )
        self.assertIn("response_schema", client.models.calls[0].get("config", {}))

    def test_mocked_client_with_text_json_succeeds(self) -> None:
        client = FakeClient(
            response=FakeResponse(
                text=json.dumps(make_outline_data()), parsed=None
            )
        )
        outline = generate_outline(**self._valid_inputs(), client=client)
        self.assertEqual(outline.title, "Test Outline")

    def test_api_error_is_wrapped(self) -> None:
        client = FakeClient(error=RuntimeError("boom"))
        with self.assertRaises(GeminiAPIError):
            generate_outline(**self._valid_inputs(), client=client)

    def test_empty_response_raises_api_error(self) -> None:
        client = FakeClient(response=FakeResponse(text="  ", parsed=None))
        with self.assertRaises(GeminiAPIError):
            generate_outline(**self._valid_inputs(), client=client)

    def test_invalid_model_output_raises_validation_error(self) -> None:
        bad = make_outline_data(
            panels=[make_outline_panel(n) for n in (1, 2, 2, 4, 5)]
        )
        client = FakeClient(response=FakeResponse(text=json.dumps(bad)))
        with self.assertRaises(OutlineValidationError):
            generate_outline(**self._valid_inputs(), client=client)

    def test_prompt_mentions_inputs_without_hardcoded_story(self) -> None:
        text = build_outline_prompt(**self._valid_inputs())
        for value in ("Nova", "Floating city", "Adventurous", "Cartoon"):
            self.assertIn(value, text)
        self.assertNotIn("fox", text.lower())


if __name__ == "__main__":
    unittest.main()
