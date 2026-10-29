"""
Test Integrity Hook
===================
Pre-commit hook that prevents AI agents (and humans) from "fixing" tests
by mocking, skipping, deleting, or weakening them instead of fixing the
actual implementation. Covers Python and JS/TS test files.

Run manually:
    python scripts/hooks/check_test_integrity.py         # staged changes
    python scripts/hooks/check_test_integrity.py <rev>   # an existing commit

As a pre-commit hook:
    Add to .pre-commit-config.yaml or run via Makefile target.

Exit codes:
    0 — all checks passed
    1 — integrity violation found

Known limits: string literals are recognized one line at a time, so text
inside a multi-line string (a triple-quoted block, a template literal
spanning lines) is still checked as code. A quote character outside a string
(an apostrophe in a comment, a quote in a JS regex literal) hides the rest of
that line. A JS test whose name is wrapped onto the next line isn't tracked
for deletion.
"""

import re
import subprocess
import sys
from collections import Counter
from pathlib import PurePosixPath

# Patterns that indicate test integrity violations
SKIP_PATTERNS = [
    r"@pytest\.mark\.skip",
    r"@pytest\.mark\.xfail",
    r"@unittest\.skip",
    r"@unittest\.skipIf",
    r"@unittest\.skipUnless",
    r"pytest\.skip\(",
    r"self\.skipTest\(",
    r"# *pragma: *no cover",  # skipping coverage = hiding failures
    r"(?<![\w.])(?:it|test|describe)(?:\.\w+)*\.(?:skip|only)\b",  # .only silently skips every other test
    r"(?<![\w.])(?:[xf]it|[xf]describe|xtest)\(",  # fit/fdescribe focus, like .only
]

MOCK_PATTERNS = [
    r"@patch\(",           # mocking at decorator level
    r"mock\.patch\(",
    r"MagicMock\(",
    r"Mock\(",
    r"mocker\.patch\(",
    r"monkeypatch\.setattr\(",
]

WEAKENED_ASSERTION_PATTERNS = [
    r"assert True$",
    r"assert 1\s*(#.*)?$",  # bare `assert 1`, but NOT assert 1 == x or assert 10 == x
    r"assert \"[^\"]*\"(\s*\))?\s*(#.*)?$",  # bare string assertion (always truthy), but NOT assert "x" in y
    r"self\.assertTrue\(True\)",
    r"assert result is not None  # relaxed",
    r"assert len\(.+\) >= 0",  # always true
    r"\.toBeTruthy\(\)",
    r"\.toBeDefined\(\)",
    r"expect\(true\)",
]

# An exact matcher replaced by a range or type check on the same subject,
# e.g. expect(x).toBe(48) -> expect(x).toBeGreaterThan(0)
EXACT_MATCHER = re.compile(r"expect\((.*?)\)\.(?:toBe|toEqual|toStrictEqual)\(")
LOOSE_MATCHER = re.compile(
    r"expect\((.*?)\)\.(?:toBeGreaterThan|toBeGreaterThanOrEqual|toBeLessThan"
    r"|toBeLessThanOrEqual|toBeCloseTo|(?:toBe|toEqual|toStrictEqual)\(expect\.any(?:thing)?)\("
)

# Group 1 is the indent, the group named "name" is the test's name
PY_TEST_HEADER = re.compile(r"(\s*)(?:async\s+)?(?:def (?=test_)|class (?=Test))(?P<name>\w+)")
JS_TEST_HEADER = re.compile(r"""(\s*)[xf]?(?:it|test|describe)(?:\.\w+)*(?:\.each\(.*?\))?\(\s*(['"`])(?P<name>.*?)\2""")


# --- Test files and their tests ------------------------------------------------

JS_SUFFIXES = (".js", ".jsx", ".ts", ".tsx")


def is_code_file(path: str) -> bool:
    return PurePosixPath(path).suffix in (".py", *JS_SUFFIXES)


def is_test_file(path: str) -> bool:
    p = PurePosixPath(path)
    if p.suffix == ".py":
        return p.name.startswith("test_") or p.name.endswith("_test.py") or "tests" in p.parts
    if p.suffix in JS_SUFFIXES:
        return bool(re.search(r"\.(test|spec)\.[jt]sx?$", p.name)) or "__tests__" in p.parts
    return False


def extract_tests(path: str, text: str) -> list[tuple[str, str]]:
    """(name, body) for each test in a file.

    The body is the header line with the name blanked out, plus every
    following line indented deeper than the header, so a renamed test with
    unchanged code has the same body.
    """
    header_re = PY_TEST_HEADER if path.endswith(".py") else JS_TEST_HEADER
    lines = text.split("\n")
    tests = []
    for i, line in enumerate(lines):
        m = header_re.match(line)
        if not m:
            continue
        indent = len(m.group(1))
        name = m.group("name")
        body = [line[:m.start("name")] + line[m.end("name"):]]
        for following in lines[i + 1:]:
            if following.strip() and len(following) - len(following.lstrip()) <= indent:
                break
            body.append(following)
        tests.append((name, "\n".join(b.strip() for b in body).strip()))
    return tests


def find_deleted_tests(old_files: dict[str, str], new_files: dict[str, str]) -> list[str]:
    """Tests present in the old files but not the new, minus renames.

    A test whose old body reappears under a name that wasn't there before
    (in any changed file, so moves between files count) was renamed.
    """
    old = [(path, name, body) for path, text in old_files.items() for name, body in extract_tests(path, text)]
    new = [(path, name, body) for path, text in new_files.items() for name, body in extract_tests(path, text)]

    missing = Counter((p, n) for p, n, _ in old) - Counter((p, n) for p, n, _ in new)
    added = Counter((p, n) for p, n, _ in new) - Counter((p, n) for p, n, _ in old)
    added_bodies = Counter(b for p, n, b in new if (p, n) in added)

    violations = []
    for path, name, body in old:
        if not missing[(path, name)]:
            continue
        missing[(path, name)] -= 1
        if added_bodies[body]:
            added_bodies[body] -= 1
            continue
        violations.append(f"Test deleted: {name} ({path})")
    return violations


# --- Diff lines ----------------------------------------------------------------------

def code_only(line: str) -> str:
    """The line with string-literal contents removed; the quotes stay.

    `write_text("def test_x(): assert 1")` becomes `write_text("")`.
    """
    out = []
    quote = None
    i = 0
    while i < len(line):
        c = line[i]
        if quote:
            if c == "\\":
                i += 1
            elif c == quote:
                quote = None
                out.append(c)
        else:
            if c in "'\"`":
                quote = c
            out.append(c)
        i += 1
    return "".join(out)


def split_diff_by_file(diff: str) -> dict[str, str]:
    """{path: that file's part of the diff}, keyed by the new path (old for deletions)."""
    files = {}
    for part in re.split(r"^diff --git ", diff, flags=re.MULTILINE)[1:]:
        m = re.match(r"a/(.*?) b/(.*?)$", part.split("\n", 1)[0])
        if m:
            files[m.group(2)] = part
    return files


def _changed_lines(diff: str, sign: str) -> list[str]:
    return [line[1:] for line in diff.split("\n")
            if line.startswith(sign) and not line.startswith(sign * 3)]


def check_for_new_skips(diff: str) -> list[str]:
    """Check if new skip/disable patterns were added."""
    violations = []
    for line in _changed_lines(diff, "+"):
        for pattern in SKIP_PATTERNS:
            if re.search(pattern, code_only(line), re.IGNORECASE):
                violations.append(f"New test skip/disable found: {line.strip()}")
    return violations


def check_for_mock_only_tests(diff: str) -> list[str]:
    """Check if new tests rely only on mocks (no real assertions)."""
    violations = []
    in_new_test = False
    test_name = ""
    has_assertion = False
    has_mock = False

    for line in map(code_only, _changed_lines(diff, "+")):
        # Detect new test function
        if re.match(r"\s*def (test_\w+)", line):
            # Check previous test if it was mock-only
            if in_new_test and has_mock and not has_assertion:
                violations.append(f"Test '{test_name}' uses mocks but has no real assertions")
            test_name = re.match(r"\s*def (test_\w+)", line).group(1)
            in_new_test = True
            has_assertion = False
            has_mock = False
            continue

        if in_new_test:
            # Check for assertions: a bare pytest `assert`, or assertEqual( / assert_called_with( etc.
            if re.search(r"\bassert\b|\bassert\w*\(", line):
                has_assertion = True
            # Check for mocks
            for pattern in MOCK_PATTERNS:
                if re.search(pattern, line):
                    has_mock = True
                    break

    # Check last test
    if in_new_test and has_mock and not has_assertion:
        violations.append(f"Test '{test_name}' uses mocks but has no real assertions")

    return violations


def check_for_weakened_assertions(diff: str) -> list[str]:
    """Check if assertions were weakened (made to always pass, or loosened)."""
    violations = []
    def subject(m: re.Match) -> str:
        return re.sub(r"\s", "", m.group(1))

    exact_subjects = {subject(m) for line in _changed_lines(diff, "-")
                      for m in EXACT_MATCHER.finditer(code_only(line))}
    for line in _changed_lines(diff, "+"):
        code = code_only(line)
        if any(re.search(p, code, re.IGNORECASE) for p in WEAKENED_ASSERTION_PATTERNS):
            violations.append(f"Weakened assertion found: {line.strip()}")
        elif any(subject(m) in exact_subjects for m in LOOSE_MATCHER.finditer(code)):
            violations.append(f"Exact assertion loosened: {line.strip()}")
    return violations


# --- Git ---------------------------------------------------------------------------

def _git(*args: str) -> str | None:
    result = subprocess.run(["git", *args], capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    return result.stdout if result.returncode == 0 else None


def get_diff(rev: str | None) -> str:
    """The staged diff, or the diff a commit introduced (a merge: against its first parent)."""
    args = ["diff", "--cached"] if rev is None else ["show", "--format=", "--diff-merges=first-parent", rev]
    return _git(*args, "--no-renames") or ""


def get_file_versions(paths: list[str], rev: str | None) -> tuple[dict[str, str], dict[str, str]]:
    """({path: text before}, {path: text after}); a missing side reads as empty."""
    old_ref, new_ref = ("HEAD:", ":") if rev is None else (f"{rev}^:", f"{rev}:")
    old = {p: _git("show", old_ref + p) or "" for p in paths}
    new = {p: _git("show", new_ref + p) or "" for p in paths}
    return old, new


def main():
    """Run all test integrity checks."""
    rev = sys.argv[1] if len(sys.argv) > 1 else None
    print("Checking test integrity...")

    if rev is not None and _git("rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}") is None:
        print(f"X Unknown revision: {rev}")
        return 1

    diff = get_diff(rev)
    if not diff:
        print("No staged changes found." if rev is None else f"No changes in {rev}.")
        return 0

    file_diffs = split_diff_by_file(diff)
    test_diffs = {p: d for p, d in file_diffs.items() if is_test_file(p)}
    old_files, new_files = get_file_versions(list(test_diffs), rev)

    all_violations = find_deleted_tests(old_files, new_files)
    # Skips (like "pragma: no cover") hide failures from source files too
    for path, file_diff in file_diffs.items():
        if is_code_file(path):
            all_violations.extend(check_for_new_skips(file_diff))
    for file_diff in test_diffs.values():
        all_violations.extend(check_for_mock_only_tests(file_diff))
        all_violations.extend(check_for_weakened_assertions(file_diff))

    if all_violations:
        print("\nX TEST INTEGRITY VIOLATIONS FOUND:")
        print("=" * 50)
        for v in all_violations:
            print(f"  - {v}")
        print("=" * 50)
        print("\nTests should be fixed by fixing the implementation, not by:")
        print("  - Skipping or disabling tests")
        print("  - Mocking away the behavior being tested")
        print("  - Deleting tests that fail")
        print("  - Weakening assertions to match buggy behavior")
        print("\nIf a test is genuinely wrong (not the implementation),")
        print("explain why in the commit message and get human review.")
        return 1

    print("OK All test integrity checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
