"""Run one unittest file and emit a machine-readable summary."""

import importlib.util
import io
import json
import sys
import unittest
from pathlib import Path


RESULT_PREFIX = "CODING_AGENT_TEST_RESULT="


def main(argv: list[str] | None = None) -> int:
    arguments = argv if argv is not None else sys.argv[1:]
    if len(arguments) != 1:
        return _emit("error", tests_run=0, skipped=0, details="expected test path")

    test_path = Path(arguments[0]).resolve()
    sys.argv = [sys.argv[0]]
    try:
        spec = importlib.util.spec_from_file_location("hidden_test_module", test_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load test module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except BaseException as error:
        return _emit(
            "error",
            tests_run=0,
            skipped=0,
            details=f"{type(error).__name__}: {error}",
        )

    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    if result.testsRun == 0:
        status = "empty"
    elif result.skipped:
        status = "skipped"
    elif result.failures or result.errors:
        status = "failed"
    else:
        status = "passed"
    return _emit(
        status,
        tests_run=result.testsRun,
        skipped=len(result.skipped),
        details=stream.getvalue(),
    )


def _emit(status: str, *, tests_run: int, skipped: int, details: str) -> int:
    payload = {
        "status": status,
        "tests_run": tests_run,
        "skipped": skipped,
        "details": details,
    }
    print(RESULT_PREFIX + json.dumps(payload, sort_keys=True))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
