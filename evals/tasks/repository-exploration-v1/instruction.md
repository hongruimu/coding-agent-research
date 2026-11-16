# Repository Exploration Task

Inspect the provided workspace without modifying its existing files.

Create `final_answer.json` with this structure:

```json
{
  "schema": "coding-agent/repository-exploration-answer-v1",
  "task": {
    "id": "repository-exploration-v1",
    "version": "1.0.0"
  },
  "facts": [
    {
      "id": "fact identifier",
      "value": "discovered value",
      "evidence": ["workspace-relative/path"]
    }
  ]
}
```

Return exactly one fact for each identifier below:

- `python_requirement`: the declared Python version requirement.
- `package_directory`: the import package directory under the source layout.
- `tested_callable`: the callable directly exercised by the committed tests.
- `test_framework`: the framework configured and imported by those tests.

Evidence paths must use POSIX separators, remain relative to the workspace, and point to files that support the value. Do not include grader paths, hidden files, absolute paths, or `..` components.
