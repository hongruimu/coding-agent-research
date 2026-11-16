import unittest

from text_tools import slugify


class SlugifyRegressionTests(unittest.TestCase):
    def test_single_space_is_replaced(self) -> None:
        self.assertEqual(slugify("Hello World"), "hello-world")

    def test_surrounding_spaces_are_removed(self) -> None:
        self.assertEqual(slugify("  Release Notes  "), "release-notes")
