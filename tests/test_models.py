"""Tests for the canonical Comic/Panel Pydantic models.

Run with the project .venv, no extra dependencies required:
    python -m unittest discover -s tests -v
"""

import unittest

from pydantic import ValidationError

from app.models import Comic, Panel


def make_panel(panel_number: int, **overrides) -> dict:
    data: dict = {
        "panel_number": panel_number,
        "title": f"Panel {panel_number}",
        "scene_description": f"Scene for panel {panel_number}",
        "narration": f"Narration for panel {panel_number}",
        "dialogue": f"Dialogue for panel {panel_number}",
        "image_prompt": f"Image prompt for panel {panel_number}",
        "image_path": None,
    }
    data.update(overrides)
    return data


def make_comic_data(**overrides) -> dict:
    data: dict = {
        "title": "Test Comic",
        "prompt": "A brave explorer discovers a floating city",
        "character": "Nova",
        "setting": "Floating city above the clouds",
        "tone": "Adventurous",
        "art_style": "Cartoon",
        "panels": [make_panel(i) for i in (1, 2, 3, 4, 5)],
    }
    data.update(overrides)
    return data


class TestComicModels(unittest.TestCase):
    def test_valid_comic_with_five_panels_succeeds(self) -> None:
        comic = Comic.model_validate(make_comic_data())
        self.assertEqual(len(comic.panels), 5)
        self.assertEqual(
            [p.panel_number for p in comic.panels], [1, 2, 3, 4, 5]
        )
        # Must round-trip through dict/JSON cleanly.
        dumped = comic.model_dump()
        self.assertEqual(len(dumped["panels"]), 5)
        Comic.model_validate(dumped)
        Comic.model_validate_json(comic.model_dump_json())

    def test_comic_with_fewer_than_five_panels_fails(self) -> None:
        data = make_comic_data(panels=[make_panel(i) for i in (1, 2, 3, 4)])
        with self.assertRaises(ValidationError):
            Comic.model_validate(data)

    def test_comic_with_more_than_five_panels_fails(self) -> None:
        data = make_comic_data(
            panels=[make_panel(i) for i in (1, 2, 3, 4, 5, 6)]
        )
        with self.assertRaises(ValidationError):
            Comic.model_validate(data)

    def test_duplicate_panel_numbers_fail(self) -> None:
        data = make_comic_data(
            panels=[make_panel(n) for n in (1, 2, 2, 4, 5)]
        )
        with self.assertRaises(ValidationError):
            Comic.model_validate(data)

    def test_missing_panel_number_or_wrong_order_fails(self) -> None:
        missing = make_comic_data(
            panels=[make_panel(n) for n in (1, 2, 3, 4, 6)]
        )
        with self.assertRaises(ValidationError):
            Comic.model_validate(missing)

        out_of_order = make_comic_data(
            panels=[make_panel(n) for n in (5, 4, 3, 2, 1)]
        )
        with self.assertRaises(ValidationError):
            Comic.model_validate(out_of_order)

    def test_empty_required_text_fields_fail(self) -> None:
        for field in (
            "title",
            "prompt",
            "character",
            "setting",
            "tone",
            "art_style",
        ):
            with self.subTest(comic_field=field):
                data = make_comic_data(**{field: "   "})
                with self.assertRaises(ValidationError):
                    Comic.model_validate(data)

        with self.subTest(panel_field="title"):
            bad_panels = [make_panel(i) for i in (1, 2, 3, 4, 5)]
            bad_panels[0]["title"] = "  "
            with self.assertRaises(ValidationError):
                Comic.model_validate(make_comic_data(panels=bad_panels))

        with self.subTest(panel_number="zero"):
            with self.assertRaises(ValidationError):
                Panel.model_validate(make_panel(0))

    def test_image_path_none_is_accepted(self) -> None:
        panel = Panel.model_validate(make_panel(1, image_path=None))
        self.assertIsNone(panel.image_path)

        comic = Comic.model_validate(make_comic_data())
        self.assertTrue(all(p.image_path is None for p in comic.panels))


if __name__ == "__main__":
    unittest.main()
