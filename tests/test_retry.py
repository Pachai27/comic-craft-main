"""Tests for bounded transient-error retry (no real API calls).

Run with the project .venv:
    python -m unittest discover -s tests -v
"""

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from app.config import get_max_attempts
from app.gemini_flash import (
    ComicOutline,
    GeminiAPIError,
    OutlineValidationError,
    generate_outline,
)
from app.gemini_pro import DetailedComic, generate_detailed_comic
from app.image_generator import ImageGenerationError, generate_image
from app.retry import call_with_retry, is_transient, status_code_of


class CodeError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"status {code}")
        self.code = code


class StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class ResponseError(Exception):
    """Mimics huggingface_hub HfHubHTTPError's response shape."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.response = SimpleNamespace(status_code=status_code)


class TestTransientClassification(unittest.TestCase):
    def test_transient_codes(self) -> None:
        for code in (429, 500, 502, 503, 599):
            with self.subTest(code=code):
                self.assertTrue(is_transient(CodeError(code)))
                self.assertTrue(is_transient(StatusError(code)))
                self.assertTrue(is_transient(ResponseError(code)))

    def test_non_transient_codes(self) -> None:
        for code in (200, 400, 401, 403, 404, 422):
            with self.subTest(code=code):
                self.assertFalse(is_transient(CodeError(code)))

    def test_errors_without_status_never_retry(self) -> None:
        self.assertFalse(is_transient(RuntimeError("boom")))
        self.assertFalse(is_transient(ValueError("bad input")))
        self.assertIsNone(status_code_of(RuntimeError("boom")))


class TestCallWithRetry(unittest.TestCase):
    def test_success_first_try_calls_once(self) -> None:
        calls: list[int] = []
        result = call_with_retry(lambda: calls.append(1) or "ok")
        self.assertEqual(result, "ok")
        self.assertEqual(len(calls), 1)

    def test_transient_then_success(self) -> None:
        calls: list[int] = []
        attempts = iter([CodeError(503), "recovered"])

        def func():
            calls.append(1)
            outcome = next(attempts)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        self.assertEqual(call_with_retry(func), "recovered")
        self.assertEqual(len(calls), 2)

    def test_second_failure_raises_after_two_attempts(self) -> None:
        calls: list[int] = []

        def func():
            calls.append(1)
            raise CodeError(503)

        with self.assertRaises(CodeError):
            call_with_retry(func, max_attempts=99)
        self.assertEqual(len(calls), 2)

    def test_non_transient_never_retried(self) -> None:
        calls: list[int] = []

        def func():
            calls.append(1)
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            call_with_retry(func)
        self.assertEqual(len(calls), 1)

    def test_single_attempt_mode(self) -> None:
        calls: list[int] = []

        def func():
            calls.append(1)
            raise CodeError(503)

        with self.assertRaises(CodeError):
            call_with_retry(func, max_attempts=1)
        self.assertEqual(len(calls), 1)


class TestMaxAttemptsConfig(unittest.TestCase):
    def test_defaults_and_clamping(self) -> None:
        cases = [
            (None, 2),
            ("", 2),
            ("   ", 2),
            ("abc", 2),
            ("1", 1),
            ("2", 2),
            ("99", 2),
            ("0", 1),
            ("-3", 1),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                with patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("AI_MAX_ATTEMPTS", None)
                    if raw is not None:
                        os.environ["AI_MAX_ATTEMPTS"] = raw
                    self.assertEqual(get_max_attempts(), expected)


def make_outline_data() -> dict:
    return {
        "title": "Retry Outline",
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
        "title": "Retry Detailed",
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


class FakeResponse:
    def __init__(self, text: str | None = None) -> None:
        self.text = text
        self.parsed = None


class ScriptedModels:
    """Fake SDK models endpoint replaying queued outcomes."""

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


class ScriptedImageClient:
    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def text_to_image(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def flash_inputs(**overrides) -> dict:
    data: dict = {
        "prompt": "A brave fox finds a glowing key",
        "character": "Ravi the fox",
        "setting": "enchanted forest",
        "tone": "dramatic",
        "art_style": "comic book",
    }
    data.update(overrides)
    return data


class TestFlashRetry(unittest.TestCase):
    def test_phase10_503_shape_retries_once_then_succeeds(self) -> None:
        from google.genai.errors import ServerError

        transient = ServerError(
            503,
            {"error": {"code": 503, "status": "UNAVAILABLE"}},
            None,
        )
        assert is_transient(transient)
        client = ScriptedClient(
            [transient, FakeResponse(text=json.dumps(make_outline_data()))]
        )
        outline = generate_outline(**flash_inputs(), client=client)
        self.assertIsInstance(outline, ComicOutline)
        self.assertEqual(
            [p.panel_number for p in outline.panels], [1, 2, 3, 4, 5]
        )
        self.assertEqual(len(client.models.calls), 2)

    def test_repeated_transient_failure_raises(self) -> None:
        from google.genai.errors import ServerError

        outcomes = [
            ServerError(503, {}, None),
            ServerError(503, {}, None),
        ]
        client = ScriptedClient(outcomes)
        with self.assertRaises(GeminiAPIError):
            generate_outline(**flash_inputs(), client=client)
        self.assertEqual(len(client.models.calls), 2)

    def test_non_transient_error_not_retried(self) -> None:
        client = ScriptedClient([RuntimeError("boom")])
        with self.assertRaises(GeminiAPIError):
            generate_outline(**flash_inputs(), client=client)
        self.assertEqual(len(client.models.calls), 1)

    def test_malformed_output_not_retried(self) -> None:
        bad = make_outline_data()
        bad["panels"] = bad["panels"][:4]
        client = ScriptedClient([FakeResponse(text=json.dumps(bad))])
        with self.assertRaises(OutlineValidationError):
            generate_outline(**flash_inputs(), client=client)
        self.assertEqual(len(client.models.calls), 1)


class TestProRetry(unittest.TestCase):
    def test_429_retries_once_then_succeeds(self) -> None:
        from google.genai.errors import ClientError

        transient = ClientError(429, {"error": {"status": "RESOURCE_EXHAUSTED"}}, None)
        assert is_transient(transient)
        from app.gemini_flash import ComicOutline as Outline

        outline = Outline.model_validate(
            {
                "title": "Outline",
                "panels": [
                    {
                        "panel_number": i,
                        "title": f"O{i}",
                        "scene_description": f"S{i}",
                    }
                    for i in (1, 2, 3, 4, 5)
                ],
            }
        )
        client = ScriptedClient(
            [transient, FakeResponse(text=json.dumps(make_detailed_data()))]
        )
        detailed = generate_detailed_comic(
            outline, **flash_inputs(), client=client
        )
        self.assertIsInstance(detailed, DetailedComic)
        self.assertEqual(len(client.models.calls), 2)

    def test_repeated_transient_failure_raises(self) -> None:
        from google.genai.errors import ServerError
        from app.gemini_flash import ComicOutline as Outline

        outline = Outline.model_validate(make_outline_data())
        client = ScriptedClient(
            [ServerError(500, {}, None), ServerError(500, {}, None)]
        )
        from app.gemini_pro import GeminiAPIError as ProAPIError

        with self.assertRaises(ProAPIError):
            generate_detailed_comic(
                outline, **flash_inputs(), client=client
            )
        self.assertEqual(len(client.models.calls), 2)


class TestImageRetry(unittest.TestCase):
    def test_transient_then_success(self) -> None:
        expected = Image.new("RGB", (16, 16), color="red")
        client = ScriptedImageClient([ResponseError(503), expected])
        result = generate_image("A fox", client=client)
        self.assertIs(result, expected)
        self.assertEqual(len(client.calls), 2)

    def test_non_transient_not_retried(self) -> None:
        client = ScriptedImageClient([RuntimeError("boom")])
        with self.assertRaises(ImageGenerationError):
            generate_image("A fox", client=client)
        self.assertEqual(len(client.calls), 1)


if __name__ == "__main__":
    unittest.main()
