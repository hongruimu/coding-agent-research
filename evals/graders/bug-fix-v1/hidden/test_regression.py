import unittest

from text_tools import slugify


class SlugifyHiddenRegressionTests(unittest.TestCase):
    def test_empty_value_remains_empty(self) -> None:
        self.assertEqual(slugify(""), "")

    def test_already_normalized_value_is_unchanged(self) -> None:
        self.assertEqual(slugify("release-notes"), "release-notes")
