"""
Run Associated Tests Hook (PostToolUse)
========================================
After Claude edits a source file, automatically runs the corresponding
test file. If tests fail, exits 2 so Claude sees the output and
can fix the implementation.

Source-to-test mapping:
    TypeScript: dir/X.ts  ->  dir/__tests__/X.test.ts, else X.test.ts (co-located),
                run with vitest from the repo root

Exit codes:
    0 — tests passed or no associated test found
    2 — tests failed (stderr is fed back to Claude; other non-zero codes
        only show up in verbose mode)
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

# Map source file patterns to their test file locations
MAPPINGS = [
    # TypeScript: dir/X.ts -> dir/__tests__/X.test.ts (components/, lib/)
    {
        "source_pattern": r"^(?P<dir>(?:.+/)?)(?P<name>[^/]+)\.(?P<ext>tsx?)$",
        "test_template": "{dir}__tests__/{name}.test.{ext}",
        "runner": "vitest",
    },
    # TypeScript: co-located X.ts -> X.test.ts
    {
        "source_pattern": r"^(?P<name>.+)\.(?P<ext>tsx?)$",
        "test_template": "{name}.test.{ext}",
        "runner": "vitest",
    },
]


def find_test_file(source_path: str) -> tuple[str | None, str | None]:
    """Map a source file to its test file. Returns (test_path, runner) or (None, None)."""
    import re

    for mapping in MAPPINGS:
        match = re.match(mapping["source_pattern"], source_path)
        if not match:
            continue

        test_path = mapping["test_template"].format(**match.groupdict())

        if Path(test_path).exists():
            return test_path, mapping["runner"]

    return None, None


def get_changed_source_files() -> list[str]:
    """Get source files with uncommitted changes (not test files)."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        files = result.stdout.strip().split("\n")
        return [f for f in files if f and not _is_test_file(f)]
    except subprocess.CalledProcessError:
        return []


def _is_test_file(path: str) -> bool:
    """Quick check if a path looks like a test file."""
    name = Path(path).name
    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or ".test." in name
        or ".spec." in name
    )


def run_tests(test_path: str, runner: str) -> tuple[bool, str]:
    """Run tests and return (success, output)."""
    if runner == "vitest":
        # shutil.which finds npx.cmd on Windows; a bare "npx" fails without a shell.
        cmd = [shutil.which("npx"), "vitest", "run", test_path]
    else:
        return True, f"Unknown runner: {runner}"

    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    output = result.stdout + result.stderr
    return result.returncode == 0, output


def main():
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0

    # Extract file path from tool input
    tool_input = data.get("tool_input", {})
    file_path = tool_input.get("file_path", "") or tool_input.get("path", "")

    if not file_path:
        return 0

    # Claude Code sends absolute paths; the mappings expect repo-relative ones.
    # Hooks run from the project root, so that's what we resolve against.
    path = Path(file_path)
    if path.is_absolute():
        try:
            path = path.relative_to(Path.cwd())
        except ValueError:
            return 0  # Outside the project, nothing to test
    file_path = path.as_posix()

    # Skip if this is itself a test file
    if _is_test_file(file_path):
        return 0

    test_path, runner = find_test_file(file_path)
    if not test_path:
        return 0  # No associated test, nothing to do

    success, output = run_tests(test_path, runner)

    if not success:
        print(
            f"TESTS FAILED for {test_path} (triggered by edit to {file_path}):\n\n{output}",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
