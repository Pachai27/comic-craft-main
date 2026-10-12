"""Tests for hosted Hugging Face image generation (no real API calls).

Run with the project .venv:
    python -m unittest discover -s tests -v
"""

import unittest
from unittest.mock import patch

from PIL import Image

from app import image_generator
from app.config import DEFAULT_HF_IMAGE_MODEL, get_hf_image_model
from app.image_generator import (
    ImageConfigError,
    ImageGenerationError,
    ImageValidationError,
    generate_image,
)


class FakeClient:
    """Mocked InferenceClient surface used to avoid real API calls."""

    def __init__(self, image=None, error: Exception | None = None) -> None:
        self._image = image
        self._error = error
        self.calls: list[dict] = []

    def text_to_image(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        if self._error is not None:
            raise self._error
        return self._image


def make_image() -> Image.Image:
    return Image.new("RGB", (64, 64), color="red")


class TestGenerateImage(unittest.TestCase):
    def test_blank_prompt_rejected(self) -> None:
        client = FakeClient(image=make_image())
        with self.assertRaises(ImageValidationError):
            generate_image("", client=client)
        with self.assertRaises(ImageValidationError):
            generate_image("   ", client=client)
        self.assertEqual(client.calls, [])

    def test_missing_token_raises_config_error(self) -> None:
        with patch(
            "app.image_generator.get_hf_token", return_value=None
        ):
            with self.assertRaises(ImageConfigError):
                generate_image("A brave explorer", client=None)

    def test_success_returns_expected_image(self) -> None:
        expected = make_image()
        client = FakeClient(image=expected)
        result = generate_image("A floating city", client=client)
        self.assertIs(result, expected)
        self.assertIsInstance(result, Image.Image)

    def test_api_failure_is_wrapped(self) -> None:
        client = FakeClient(error=RuntimeError("boom"))
        with self.assertRaises(ImageGenerationError):
            generate_image("A floating city", client=client)

    def test_configured_model_passed_through(self) -> None:
        client = FakeClient(image=make_image())
        generate_image("A floating city", client=client)
        self.assertEqual(client.calls[0].get("model"), get_hf_image_model())
        self.assertEqual(
            get_hf_image_model(), DEFAULT_HF_IMAGE_MODEL
        )

        client2 = FakeClient(image=make_image())
        generate_image(
            "A floating city", client=client2, model="custom/model-id"
        )
        self.assertEqual(client2.calls[0].get("model"), "custom/model-id")

    def test_provider_auto_is_used(self) -> None:
        created: dict = {}

        class RecordingClient:
            def __init__(self, *args, **kwargs) -> None:
                created.update(kwargs)
                created["args"] = args
                self._inner = FakeClient(image=make_image())

            def text_to_image(self, prompt: str, **kwargs):
                return self._inner.text_to_image(prompt, **kwargs)

        with patch(
            "huggingface_hub.InferenceClient", RecordingClient
        ):
            result = generate_image("A floating city", api_key="fake-token")
        self.assertEqual(created.get("provider"), "auto")
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(image_generator.PROVIDER, "auto")


if __name__ == "__main__":
    unittest.main()
