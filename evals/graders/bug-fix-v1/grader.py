"""Deterministic task-specific grader for the bug-fix v1 fixture."""

import argparse
import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any


CRITERIA_SCHEMA = "coding-agent/bug-fix-criteria-v1"
RESULT_PREFIX = "CODING_AGENT_TEST_RESULT="
IGNORED_DIRECTORY_NAMES = {".pytest_cache", "__pycache__"}


class GraderConfigurationError(ValueError):
    """Raised when committed grader assets are malformed or modified."""


def grade_candidate(
    candidate_workspace: str | Path,
    baseline_workspace: str | Path,
    criteria_path: str | Path,
) -> dict[str, object]:
    """Grade one candidate workspace against the fixed bug-fix baseline."""

    baseline = _require_directory(Path(baseline_workspace), "baseline workspace")
    candidate = _require_directory(Path(candidate_workspace), "candidate workspace")
    criteria = _load_criteria(Path(criteria_path), baseline)

    candidate_issue = _validate_candidate_tree(candidate, criteria["limits"])
    if candidate_issue is not None:
        return _result(False, [candidate_issue], [], {})

    baseline_snapshot = _snapshot(baseline)
    candidate_snapshot = _snapshot(candidate)
    changed_files = sorted(
        path
        for path in set(baseline_snapshot) | set(candidate_snapshot)
        if baseline_snapshot.get(path) != candidate_snapshot.get(path)
    )
    failure_reasons: list[str] = []
    if not changed_files:
        failure_reasons.append("diff:empty")
    allowed_changes = set(criteria["allowed_changes"])
    outside_allowed = [
        path for path in changed_files if path not in allowed_changes
    ]
    if outside_allowed:
        failure_reasons.extend(
            f"diff:outside_allowed_paths:{path}" for path in outside_allowed
        )
        return _result(False, failure_reasons, changed_files, {})

    test_results: dict[str, dict[str, object]] = {}
    with tempfile.TemporaryDirectory(prefix="coding-agent-bug-fix-") as temp_dir:
        for index, group in enumerate(criteria["test_groups"]):
            group_root = Path(temp_dir) / f"group-{index}"
            run_workspace = group_root / "workspace"
            grader_assets = group_root / "grader-assets"
            shutil.copytree(candidate, run_workspace, ignore=_copy_ignore)
            grader_assets.mkdir(parents=True)
            runner_path = grader_assets / "runner.py"
            shutil.copy2(criteria["runner_path"], runner_path)
            if group["source"] == "workspace":
                test_path = run_workspace / group["path"]
            else:
                test_path = grader_assets / "case.py"
                shutil.copy2(criteria["grader_root"] / group["path"], test_path)
            details = _run_test_group(
                run_workspace=run_workspace,
                runner_path=runner_path,
                test_path=test_path,
                timeout_seconds=criteria["limits"]["timeout_seconds_per_group"],
                max_output_bytes=criteria["limits"]["max_output_bytes_per_stream"],
            )
            group_id = group["id"]
            test_results[group_id] = details
            if details["status"] != "passed":
                failure_reasons.append(
                    f"tests:{group_id}:{details['status']}"
                )

    return _result(
        not failure_reasons,
        failure_reasons,
        changed_files,
        test_results,
    )


def _load_criteria(path: Path, baseline: Path) -> dict[str, Any]:
    grader_root = path.resolve().parent.parent
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GraderConfigurationError("cannot read bug-fix criteria") from error
    if not isinstance(document, dict) or set(document) != {
        "schema",
        "task",
        "allowed_changes",
        "baseline_files",
        "limits",
        "test_runner",
        "test_groups",
    }:
        raise GraderConfigurationError("bug-fix criteria fields are invalid")
    if document["schema"] != CRITERIA_SCHEMA:
        raise GraderConfigurationError("bug-fix criteria schema is invalid")
    _validate_identity(document["task"])
    allowed_changes = _validate_path_list(
        document["allowed_changes"],
        "allowed_changes",
    )
    _validate_baseline_files(document["baseline_files"], baseline)
    limits = _validate_limits(document["limits"])
    runner = _validate_asset(
        document["test_runner"],
        root=grader_root,
        label="test runner",
    )
    groups = _validate_test_groups(
        document["test_groups"],
        grader_root=grader_root,
        baseline=baseline,
    )
    expected_groups = {"public_regression", "hidden_target", "hidden_regression"}
    if {group["id"] for group in groups} != expected_groups:
        raise GraderConfigurationError("test groups are incomplete")
    return {
        "allowed_changes": allowed_changes,
        "limits": limits,
        "runner_path": runner,
        "test_groups": groups,
        "grader_root": grader_root,
    }


def _validate_baseline_files(value: object, baseline: Path) -> None:
    if not isinstance(value, list) or not value:
        raise GraderConfigurationError("baseline_files must be a non-empty array")
    observed_paths: set[str] = set()
    for asset in value:
        asset_path = _validate_asset(asset, root=baseline, label="baseline file")
        relative = asset_path.relative_to(baseline).as_posix()
        if relative in observed_paths:
            raise GraderConfigurationError("baseline file paths must be unique")
        observed_paths.add(relative)
    actual_paths = set(_snapshot(baseline))
    if observed_paths != actual_paths:
        raise GraderConfigurationError("baseline file set does not match criteria")


def _validate_identity(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {"id", "version"}:
        raise GraderConfigurationError("task identity is invalid")
    if value != {"id": "bug-fix-v1", "version": "1.0.0"}:
        raise GraderConfigurationError("task identity does not match bug-fix-v1")


def _validate_limits(value: object) -> dict[str, int]:
    expected = {
        "timeout_seconds_per_group",
        "max_output_bytes_per_stream",
        "max_candidate_files",
        "max_candidate_bytes",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise GraderConfigurationError("limits are invalid")
    if any(
        not isinstance(value[key], int)
        or isinstance(value[key], bool)
        or value[key] <= 0
        for key in expected
    ):
        raise GraderConfigurationError("limits must be positive integers")
    return dict(value)


def _validate_test_groups(
    value: object,
    *,
    grader_root: Path,
    baseline: Path,
) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise GraderConfigurationError("test_groups must be a non-empty array")
    groups: list[dict[str, str]] = []
    group_ids: set[str] = set()
    for group in value:
        if not isinstance(group, dict) or set(group) != {
            "id",
            "source",
            "path",
            "sha256",
        }:
            raise GraderConfigurationError("test group fields are invalid")
        group_id = group["id"]
        source = group["source"]
        if not isinstance(group_id, str) or not group_id or group_id in group_ids:
            raise GraderConfigurationError("test group ids must be unique")
        if source not in {"workspace", "grader"}:
            raise GraderConfigurationError("test group source is invalid")
        root = baseline if source == "workspace" else grader_root
        asset_path = _validate_asset(group, root=root, label=f"test group {group_id}")
        group_ids.add(group_id)
        groups.append(
            {
                "id": group_id,
                "source": source,
                "path": asset_path.relative_to(root).as_posix(),
            }
        )
    return groups


def _validate_asset(value: object, *, root: Path, label: str) -> Path:
    if not isinstance(value, dict):
        raise GraderConfigurationError(f"{label} must be an object")
    path_value = value.get("path")
    digest = value.get("sha256")
    if not isinstance(path_value, str) or not isinstance(digest, str):
        raise GraderConfigurationError(f"{label} path and sha256 are required")
    relative = _relative_path(path_value, label)
    asset_path = (root / Path(*relative.parts)).resolve()
    if not asset_path.is_relative_to(root.resolve()) or not asset_path.is_file():
        raise GraderConfigurationError(f"{label} path is invalid")
    actual_digest = hashlib.sha256(asset_path.read_bytes()).hexdigest()
    if actual_digest != digest:
        raise GraderConfigurationError(f"{label} integrity mismatch")
    return asset_path


def _validate_path_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise GraderConfigurationError(f"{label} must be a non-empty array")
    paths = [_relative_path(item, label).as_posix() for item in value]
    if len(paths) != len(set(paths)):
        raise GraderConfigurationError(f"{label} paths must be unique")
    return paths


def _relative_path(value: object, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value.strip() or "\\" in value:
        raise GraderConfigurationError(f"{label} path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or path == PurePosixPath(".") or ".." in path.parts:
        raise GraderConfigurationError(f"{label} path is invalid")
    return path


def _validate_candidate_tree(root: Path, limits: dict[str, int]) -> str | None:
    file_count = 0
    total_bytes = 0
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            return f"candidate:symlink:{relative}"
        if not path.is_dir() and not path.is_file():
            return f"candidate:special_file:{relative}"
        if path.is_file() and not _is_ignored(path.relative_to(root)):
            file_count += 1
            total_bytes += path.stat().st_size
    if file_count > limits["max_candidate_files"]:
        return "candidate:file_limit_exceeded"
    if total_bytes > limits["max_candidate_bytes"]:
        return "candidate:byte_limit_exceeded"
    return None


def _snapshot(root: Path) -> dict[str, bytes]:
    snapshot: dict[str, bytes] = {}
    for path in root.rglob("*"):
        if path.is_file() and not _is_ignored(path.relative_to(root)):
            snapshot[path.relative_to(root).as_posix()] = path.read_bytes()
    return snapshot


def _is_ignored(relative: Path) -> bool:
    return (
        any(part in IGNORED_DIRECTORY_NAMES for part in relative.parts)
        or relative.suffix == ".pyc"
    )


def _copy_ignore(directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        path = Path(directory) / name
        if name in IGNORED_DIRECTORY_NAMES or path.suffix == ".pyc":
            ignored.add(name)
    return ignored


def _run_test_group(
    *,
    run_workspace: Path,
    runner_path: Path,
    test_path: Path,
    timeout_seconds: int,
    max_output_bytes: int,
) -> dict[str, object]:
    started = time.monotonic()
    environment = {
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(run_workspace / "src"),
    }
    for name in ("LANG", "LC_ALL", "PATH", "SYSTEMROOT", "TMPDIR"):
        if name in os.environ:
            environment[name] = os.environ[name]
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        return_code, timed_out = _run_process(
            [sys.executable, str(runner_path), str(test_path)],
            cwd=run_workspace,
            environment=environment,
            stdout_file=stdout_file,
            stderr_file=stderr_file,
            timeout_seconds=timeout_seconds,
        )
        if timed_out:
            stdout, retained_stdout = _read_output(stdout_file, max_output_bytes)
            stderr, retained_stderr = _read_output(stderr_file, max_output_bytes)
            return {
                "status": "timeout",
                "tests_run": 0,
                "skipped": 0,
                "exit_code": None,
                "duration_ms": round((time.monotonic() - started) * 1000),
                "stdout": retained_stdout,
                "stderr": retained_stderr,
            }
        stdout, retained_stdout = _read_output(stdout_file, max_output_bytes)
        stderr, retained_stderr = _read_output(stderr_file, max_output_bytes)

    if retained_stdout["truncated"] or retained_stderr["truncated"]:
        return {
            "status": "output_limit_exceeded",
            "tests_run": 0,
            "skipped": 0,
            "exit_code": return_code,
            "duration_ms": round((time.monotonic() - started) * 1000),
            "stdout": retained_stdout,
            "stderr": retained_stderr,
        }
    payload = _parse_runner_payload(stdout)
    status = "error"
    tests_run = 0
    skipped = 0
    if payload is not None:
        payload_status = payload.get("status")
        payload_tests_run = payload.get("tests_run")
        payload_skipped = payload.get("skipped")
        if (
            payload_status in {"passed", "failed", "skipped", "empty", "error"}
            and isinstance(payload_tests_run, int)
            and not isinstance(payload_tests_run, bool)
            and payload_tests_run >= 0
            and isinstance(payload_skipped, int)
            and not isinstance(payload_skipped, bool)
            and payload_skipped >= 0
        ):
            status = payload_status
            tests_run = payload_tests_run
            skipped = payload_skipped
    if (status == "passed") != (return_code == 0):
        status = "error"
    return {
        "status": status,
        "tests_run": tests_run,
        "skipped": skipped,
        "exit_code": return_code,
        "duration_ms": round((time.monotonic() - started) * 1000),
        "stdout": retained_stdout,
        "stderr": retained_stderr,
    }


def _run_process(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    stdout_file: Any,
    stderr_file: Any,
    timeout_seconds: int,
) -> tuple[int | None, bool]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdout=stdout_file,
        stderr=stderr_file,
        start_new_session=os.name == "posix",
    )
    try:
        return_code = process.wait(timeout=timeout_seconds)
        _terminate_process_group(process)
        return return_code, False
    except subprocess.TimeoutExpired:
        _terminate_process_group(process)
        process.wait()
        return None, True


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.kill()


def _parse_runner_payload(stdout: bytes) -> dict[str, object] | None:
    for line in stdout.decode("utf-8", errors="replace").splitlines():
        if line.startswith(RESULT_PREFIX):
            try:
                payload = json.loads(line.removeprefix(RESULT_PREFIX))
            except json.JSONDecodeError:
                return None
            return payload if isinstance(payload, dict) else None
    return None


def _read_output(
    stream: Any,
    max_bytes: int,
) -> tuple[bytes, dict[str, object]]:
    stream.flush()
    input_bytes = stream.seek(0, os.SEEK_END)
    stream.seek(0)
    payload = stream.read(max_bytes)
    retained = payload.decode("utf-8", errors="ignore")
    return payload, {
        "text": retained,
        "input_bytes": input_bytes,
        "stored_bytes": len(retained.encode("utf-8")),
        "truncated": input_bytes > max_bytes,
    }


def _require_directory(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise ValueError(f"{label} must be an existing directory")
    return resolved


def _result(
    passed: bool,
    failure_reasons: list[str],
    changed_files: list[str],
    tests: dict[str, dict[str, object]],
) -> dict[str, object]:
    return {
        "passed": passed,
        "failure_reasons": failure_reasons,
        "changed_files": changed_files,
        "tests": tests,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_workspace", type=Path)
    parser.add_argument("baseline_workspace", type=Path)
    parser.add_argument("criteria", type=Path)
    arguments = parser.parse_args(argv)
    result = grade_candidate(
        arguments.candidate_workspace,
        arguments.baseline_workspace,
        arguments.criteria,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
