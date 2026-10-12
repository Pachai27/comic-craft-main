"""Tests for the HF hosted text provider (no real API calls).

Run with the project .venv:
    python -m unittest discover -s tests -v
"""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from app import hf_text
from app.config import (
    DEFAULT_HF_TEXT_MODEL,
    get_hf_text_model,
    get_text_provider,
)
from app.gemini_flash import ComicOutline
from app.gemini_pro import DetailedComic
from app.hf_text import (
    HFTextConfigError,
    HFTextGenerationError,
    HFTextValidationError,
    generate_detailed_comic,
    generate_outline,
)
from app.models import Comic
from app.pipeline import generate_comic


def make_outline_data() -> dict:
    return {
        "title": "HF Outline",
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
        "title": "HF Detailed",
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


class FakeChatResponse:
    def __init__(self, content: str | None) -> None:
        self.choices = [
            SimpleNamespace(message=SimpleNamespace(content=content))
        ]


class ScriptedChatClient:
    """Fake InferenceClient surface replaying queued outcomes."""

    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class Transient503(Exception):
    def __init__(self) -> None:
        super().__init__("service unavailable")
        self.response = SimpleNamespace(status_code=503)


class TestHFOutline(unittest.TestCase):
    def test_valid_outline_succeeds(self) -> None:
        client = ScriptedChatClient(
            [FakeChatResponse(json.dumps(make_outline_data()))]
        )
        outline = generate_outline(**flash_inputs(), client=client)
        self.assertIsInstance(outline, ComicOutline)
        self.assertEqual(
            [p.panel_number for p in outline.panels], [1, 2, 3, 4, 5]
        )

    def test_fenced_json_is_accepted(self) -> None:
        text = "```json\n" + json.dumps(make_outline_data()) + "\n```"
        client = ScriptedChatClient([FakeChatResponse(text)])
        outline = generate_outline(**flash_inputs(), client=client)
        self.assertEqual(len(outline.panels), 5)

    def test_structured_output_and_model_requested(self) -> None:
        client = ScriptedChatClient(
            [FakeChatResponse(json.dumps(make_outline_data()))]
        )
        generate_outline(**flash_inputs(), client=client)
        call = client.calls[0]
        self.assertEqual(call.get("model"), DEFAULT_HF_TEXT_MODEL)
        fmt = call.get("response_format", {})
        self.assertEqual(fmt.get("type"), "json_schema")
        roles = [m["role"] for m in call.get("messages", [])]
        self.assertEqual(roles, ["system", "user"])

    def test_model_override_passed_through(self) -> None:
        client = ScriptedChatClient(
            [FakeChatResponse(json.dumps(make_outline_data()))]
        )
        generate_outline(
            **flash_inputs(), client=client, model="custom/text-model"
        )
        self.assertEqual(client.calls[0].get("model"), "custom/text-model")

    def test_malformed_output_rejected(self) -> None:
        for bad_text in (
            "not json at all",
            json.dumps([1, 2, 3]),
            json.dumps("just a string"),
        ):
            with self.subTest(bad=bad_text[:20]):
                client = ScriptedChatClient([FakeChatResponse(bad_text)])
                with self.assertRaises(HFTextValidationError):
                    generate_outline(**flash_inputs(), client=client)
                self.assertEqual(len(client.calls), 1)

    def test_schema_violating_output_rejected(self) -> None:
        bad = make_outline_data()
        bad["panels"] = bad["panels"][:4]
        client = ScriptedChatClient([FakeChatResponse(json.dumps(bad))])
        with self.assertRaises(HFTextValidationError):
            generate_outline(**flash_inputs(), client=client)

    def test_blank_input_rejected_without_call(self) -> None:
        client = ScriptedChatClient([])
        with self.assertRaises(HFTextValidationError):
            generate_outline(
                **flash_inputs(prompt="   "), client=client
            )
        self.assertEqual(client.calls, [])

    def test_missing_token_raises_config_error(self) -> None:
        with patch("app.hf_text.get_hf_token", return_value=None):
            with self.assertRaises(HFTextConfigError):
                generate_outline(**flash_inputs(), client=None)

    def test_transient_then_success(self) -> None:
        client = ScriptedChatClient(
            [
                Transient503(),
                FakeChatResponse(json.dumps(make_outline_data())),
            ]
        )
        outline = generate_outline(**flash_inputs(), client=client)
        self.assertEqual(len(outline.panels), 5)
        self.assertEqual(len(client.calls), 2)

    def test_repeated_transient_failure_raises(self) -> None:
        client = ScriptedChatClient([Transient503(), Transient503()])
        with self.assertRaises(HFTextGenerationError):
            generate_outline(**flash_inputs(), client=client)
        self.assertEqual(len(client.calls), 2)

    def test_non_transient_not_retried(self) -> None:
        client = ScriptedChatClient([RuntimeError("boom")])
        with self.assertRaises(HFTextGenerationError):
            generate_outline(**flash_inputs(), client=client)
        self.assertEqual(len(client.calls), 1)

    def test_timeout_error_wrapped_without_retry(self) -> None:
        client = ScriptedChatClient([TimeoutError("timed out")])
        with self.assertRaises(HFTextGenerationError):
            generate_outline(**flash_inputs(), client=client)
        self.assertEqual(len(client.calls), 1)


class TestHFDetailed(unittest.TestCase):
    def _outline(self) -> ComicOutline:
        return ComicOutline.model_validate(make_outline_data())

    def test_valid_detailed_succeeds(self) -> None:
        client = ScriptedChatClient(
            [FakeChatResponse(json.dumps(make_detailed_data()))]
        )
        detailed = generate_detailed_comic(
            self._outline(), **flash_inputs(), client=client
        )
        self.assertIsInstance(detailed, DetailedComic)
        self.assertEqual(len(detailed.panels), 5)
        self.assertTrue(all(p.narration for p in detailed.panels))

    def test_malformed_detailed_rejected(self) -> None:
        client = ScriptedChatClient([FakeChatResponse("nope")])
        with self.assertRaises(HFTextValidationError):
            generate_detailed_comic(
                self._outline(), **flash_inputs(), client=client
            )

    def test_bad_outline_rejected_without_call(self) -> None:
        bad = make_outline_data()
        bad["panels"] = bad["panels"][:2]
        client = ScriptedChatClient([])
        with self.assertRaises(HFTextValidationError):
            generate_detailed_comic(bad, **flash_inputs(), client=client)
        self.assertEqual(client.calls, [])

    def test_converts_to_canonical_comic(self) -> None:
        from app.gemini_pro import to_comic

        client = ScriptedChatClient(
            [FakeChatResponse(json.dumps(make_detailed_data()))]
        )
        detailed = generate_detailed_comic(
            self._outline(), **flash_inputs(), client=client
        )
        comic = to_comic(detailed, **flash_inputs())
        self.assertIsInstance(comic, Comic)
        self.assertEqual(len(comic.panels), 5)


class TestHFClientConstruction(unittest.TestCase):
    def test_timeout_and_provider_used(self) -> None:
        created: list[dict] = []

        class RecordingClient:
            def __init__(self, *args, **kwargs) -> None:
                created.append(kwargs)

            def chat_completion(self, **kwargs):
                return FakeChatResponse(json.dumps(make_outline_data()))

        with patch("huggingface_hub.InferenceClient", RecordingClient):
            generate_outline(**flash_inputs(), api_key="fake-token")
        self.assertEqual(created[0].get("provider"), "auto")
        self.assertEqual(created[0].get("timeout"), 60.0)
        self.assertEqual(hf_text.PROVIDER, "auto")


class TestHFTextConfig(unittest.TestCase):
    def test_model_default_and_override(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HF_TEXT_MODEL", None)
            self.assertEqual(get_hf_text_model(), DEFAULT_HF_TEXT_MODEL)
            os.environ["HF_TEXT_MODEL"] = "custom/model"
            self.assertEqual(get_hf_text_model(), "custom/model")

    def test_provider_default_and_values(self) -> None:
        cases = [
            (None, "hf"),
            ("", "hf"),
            ("hf", "hf"),
            ("gemini", "gemini"),
            (" HF ", "hf"),
            ("GEMINI", "gemini"),
            ("other", "hf"),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                with patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("COMIC_TEXT_PROVIDER", None)
                    if raw is not None:
                        os.environ["COMIC_TEXT_PROVIDER"] = raw
                    self.assertEqual(get_text_provider(), expected)


class TestPipelineProviderSelection(unittest.TestCase):
    def _image_fn(self, calls: list):
        def fake(prompt: str):
            calls.append(prompt)
            return Image.new("RGB", (8, 8), color="blue")

        return fake

    def test_default_provider_is_hf(self) -> None:
        image_calls: list[str] = []
        with (
            patch.dict(os.environ, {}, clear=False),
            patch("app.hf_text.generate_outline") as mock_outline,
            patch("app.hf_text.generate_detailed_comic") as mock_detail,
        ):
            os.environ.pop("COMIC_TEXT_PROVIDER", None)
            mock_outline.return_value = ComicOutline.model_validate(
                make_outline_data()
            )
            mock_detail.return_value = DetailedComic.model_validate(
                make_detailed_data()
            )
            with tempfile.TemporaryDirectory() as tmp:
                comic = generate_comic(
                    **flash_inputs(),
                    image_fn=self._image_fn(image_calls),
                    panels_base_dir=tmp,
                )
        mock_outline.assert_called_once()
        mock_detail.assert_called_once()
        self.assertEqual(len(comic.panels), 5)
        self.assertEqual(len(image_calls), 5)

    def test_gemini_provider_uses_gemini_functions(self) -> None:
        image_calls: list[str] = []
        with (
            patch.dict(os.environ, {"COMIC_TEXT_PROVIDER": "gemini"}),
            patch("app.pipeline.generate_outline") as mock_outline,
            patch("app.pipeline.generate_detailed_comic") as mock_detail,
        ):
            mock_outline.return_value = ComicOutline.model_validate(
                make_outline_data()
            )
            mock_detail.return_value = DetailedComic.model_validate(
                make_detailed_data()
            )
            with tempfile.TemporaryDirectory() as tmp:
                generate_comic(
                    **flash_inputs(),
                    image_fn=self._image_fn(image_calls),
                    panels_base_dir=tmp,
                )
        mock_outline.assert_called_once()
        mock_detail.assert_called_once()

    def test_explicit_injection_wins(self) -> None:
        from unittest.mock import MagicMock

        outline_fn = MagicMock(
            return_value=ComicOutline.model_validate(make_outline_data())
        )
        detail_fn = MagicMock(
            return_value=DetailedComic.model_validate(make_detailed_data())
        )
        with tempfile.TemporaryDirectory() as tmp:
            generate_comic(
                **flash_inputs(),
                outline_fn=outline_fn,
                detail_fn=detail_fn,
                image_fn=self._image_fn([]),
                panels_base_dir=tmp,
            )
        outline_fn.assert_called_once()
        detail_fn.assert_called_once()


if __name__ == "__main__":
    unittest.main()
