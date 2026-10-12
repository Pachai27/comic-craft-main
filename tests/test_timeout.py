"""Tests for bounded provider request timeouts (no real API calls).

Run with the project .venv:
    python -m unittest discover -s tests -v
"""

import json
import os
import unittest
from unittest.mock import patch

from PIL import Image

from app.config import get_timeout_seconds
from app.gemini_flash import GeminiAPIError, generate_outline
from app.gemini_pro import generate_detailed_comic
from app.image_generator import ImageGenerationError, generate_image


class TestTimeoutConfig(unittest.TestCase):
    def test_parsing_and_clamping(self) -> None:
        cases = [
            (None, 60.0),
            ("", 60.0),
            ("   ", 60.0),
            ("abc", 60.0),
            ("nan", 60.0),
            ("inf", 60.0),
            ("30", 30.0),
            ("1.5", 1.5),
            ("0", 1.0),
            ("-5", 1.0),
            ("9999", 300.0),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                with patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("AI_TIMEOUT_SECONDS", None)
                    if raw is not None:
                        os.environ["AI_TIMEOUT_SECONDS"] = raw
                    self.assertEqual(get_timeout_seconds(), expected)


def make_outline_data() -> dict:
    return {
        "title": "Timeout Outline",
        "panels": [
            {
                "panel_number": i,
                "title": f"Outline {i}",
                "scene_description": f"Scene {i}",
            }
            for i in (1, 2, 3, 4, 5)
        ],
    }


def make_detailed_data() -> dict:
    return {
        "title": "Timeout Detailed",
        "panels": [
            {
                "panel_number": i,
                "title": f"Panel {i}",
                "scene_description": f"Scene {i}",
                "narration": f"Narration {i}",
                "dialogue": f"Dialogue {i}",
                "image_prompt": f"Image prompt {i}",
            }
            for i in (1, 2, 3, 4, 5)
        ],
    }


def flash_inputs() -> dict:
    return {
        "prompt": "A brave fox finds a glowing key",
        "character": "Ravi the fox",
        "setting": "enchanted forest",
        "tone": "dramatic",
        "art_style": "comic book",
    }


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.parsed = None


class ScriptedModels:
    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class ScriptedClient:
    def __init__(self, outcomes: list) -> None:
        self.models = ScriptedModels(outcomes)


class TestGeminiTimeoutPropagation(unittest.TestCase):
    def test_flash_sends_timeout_in_milliseconds(self) -> None:
        client = ScriptedClient(
            [FakeResponse(text=json.dumps(make_outline_data()))]
        )
        generate_outline(**flash_inputs(), client=client)
        config = client.models.calls[0]["config"]
        self.assertEqual(config["http_options"], {"timeout": 60000})

    def test_flash_timeout_follows_env_override(self) -> None:
        client = ScriptedClient(
            [FakeResponse(text=json.dumps(make_outline_data()))]
        )
        with patch.dict(os.environ, {"AI_TIMEOUT_SECONDS": "30"}):
            generate_outline(**flash_inputs(), client=client)
        config = client.models.calls[0]["config"]
        self.assertEqual(config["http_options"], {"timeout": 30000})

    def test_pro_sends_timeout_in_milliseconds(self) -> None:
        from app.gemini_flash import ComicOutline

        outline = ComicOutline.model_validate(make_outline_data())
        client = ScriptedClient(
            [FakeResponse(text=json.dumps(make_detailed_data()))]
        )
        generate_detailed_comic(
            outline, **flash_inputs(), client=client
        )
        config = client.models.calls[0]["config"]
        self.assertEqual(config["http_options"], {"timeout": 60000})

    def test_timeout_error_not_retried_and_wrapped(self) -> None:
        client = ScriptedClient([TimeoutError("timed out")])
        with self.assertRaises(GeminiAPIError):
            generate_outline(**flash_inputs(), client=client)
        self.assertEqual(len(client.models.calls), 1)


class RecordingInferenceClient:
    created: list[dict] = []

    def __init__(self, *args, **kwargs) -> None:
        type(self).created.append(kwargs)

    def text_to_image(self, prompt: str, **kwargs):
        return Image.new("RGB", (8, 8), color="green")


class TestHFTimeoutPropagation(unittest.TestCase):
    def test_client_built_with_default_timeout(self) -> None:
        RecordingInferenceClient.created.clear()
        with patch(
            "huggingface_hub.InferenceClient", RecordingInferenceClient
        ):
            generate_image("A fox", api_key="fake-token")
        self.assertEqual(
            RecordingInferenceClient.created[0].get("timeout"), 60.0
        )
        self.assertEqual(
            RecordingInferenceClient.created[0].get("provider"), "auto"
        )

    def test_client_timeout_follows_env_override(self) -> None:
        RecordingInferenceClient.created.clear()
        with (
            patch(
                "huggingface_hub.InferenceClient", RecordingInferenceClient
            ),
            patch.dict(os.environ, {"AI_TIMEOUT_SECONDS": "45"}),
        ):
            generate_image("A fox", api_key="fake-token")
        self.assertEqual(
            RecordingInferenceClient.created[0].get("timeout"), 45.0
        )

    def test_timeout_error_not_retried_and_wrapped(self) -> None:
        calls: list[str] = []

        class FailingClient:
            def text_to_image(self, prompt: str, **kwargs):
                calls.append(prompt)
                raise TimeoutError("timed out")

        with self.assertRaises(ImageGenerationError):
            generate_image("A fox", client=FailingClient())
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
