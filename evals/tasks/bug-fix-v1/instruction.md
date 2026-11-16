# Small Bug Fix Task

The `slugify` behavior mishandles consecutive whitespace and can emit repeated separators.

Fix the implementation while preserving the existing behavior covered by the committed tests.

Constraints:

- Only modify `src/text_tools/slug.py`.
- Do not modify tests, project configuration, or hidden grader assets.
- Keep the implementation deterministic and dependency-free.
- The final workspace must pass both existing regression tests and independent hidden tests.
