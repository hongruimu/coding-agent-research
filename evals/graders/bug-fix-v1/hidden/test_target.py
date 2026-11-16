import unittest

from text_tools import slugify


class SlugifyTargetTests(unittest.TestCase):
    def test_consecutive_spaces_collapse_to_one_separator(self) -> None:
        self.assertEqual(slugify("Hello   World"), "hello-world")

    def test_mixed_whitespace_collapses_to_one_separator(self) -> None:
        self.assertEqual(slugify("Release\t  Candidate"), "release-candidate")
