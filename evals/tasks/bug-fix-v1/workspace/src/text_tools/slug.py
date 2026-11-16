"""Slug formatting behavior with one intentional whitespace bug."""


def slugify(value: str) -> str:
    return value.strip().lower().replace(" ", "-")
