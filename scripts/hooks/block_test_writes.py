"""
Block Test Writes Hook (PreToolUse)
====================================
Makes Claude ask before modifying committed test files. Tests should be fixed
by fixing the implementation, not by weakening or rewriting the tests; when a
test really is wrong (the spec or data changed), the user approves the edit.

A test file is protected once it's in HEAD. Test files created in the current,
uncommitted work stay editable, so TDD can grow one file slice by slice.
Outside a git repo, any existing test file is protected.

Handles two tool shapes:
    Write / Edit          tool_input.file_path
    Bash / PowerShell     tool_input.command — asks when a committed test
                          file is a redirect or tee target, or shares a
                          command segment with a mutating verb (sed -i, rm,
                          mv, Set-Content, ...). A heuristic guard against the
                          easy path, not a sandbox.

Output: always exit 0. For a protected path, stdout carries a PreToolUse
"ask" decision, so Claude Code shows the user the edit to approve or reject.
"""

import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

TEST_PATTERNS = [
    r"(?:^|[/\\])test_[^/\\]+\.py$",       # test_*.py
    r"[^/\\]+_test\.py$",       # *_test.py
    r"[^/\\]+\.test\.tsx?$",    # *.test.ts / *.test.tsx
    r"[^/\\]+\.spec\.tsx?$",    # *.spec.ts / *.spec.tsx
    r"tests[/\\].*\.py$",       # anything under tests/
]

# A segment containing one of these may write any path it names.
MUTATING_VERB = re.compile(
    r"^(?:rm|mv|cp|truncate|unlink|shred|dd|install)$"
    r"|^(?:set-content|add-content|out-file|clear-content|remove-item"
    r"|move-item|copy-item|rename-item|new-item|del|erase|ren|move|copy)$",
    re.IGNORECASE,
)
SEGMENT_SPLIT = re.compile(r"&&|\|\||[;|\n]")
REDIRECT = re.compile(r"^(?:\d|&)?>>?(.*)$")


def is_test_file(file_path: str) -> bool:
    """Check if a file path matches any test file pattern."""
    for pattern in TEST_PATTERNS:
        if re.search(pattern, file_path, re.IGNORECASE):
            return True
    return False


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(cwd), *args],
                          capture_output=True, text=True)


def committed_in_head(path: Path) -> bool | None:
    """Whether path is in HEAD; None when it isn't inside a git repo."""
    anchor = path.parent
    while not anchor.exists() and anchor != anchor.parent:
        anchor = anchor.parent
    top = _git(anchor, "rev-parse", "--show-toplevel")
    if top.returncode != 0:
        return None
    try:
        rel = path.resolve().relative_to(Path(top.stdout.strip()).resolve())
    except ValueError:
        return None
    return _git(anchor, "cat-file", "-e", f"HEAD:{rel.as_posix()}").returncode == 0


def is_protected(path: Path) -> bool:
    if not is_test_file(str(path)):
        return False
    committed = committed_in_head(path)
    return path.exists() if committed is None else committed


def _tokens(segment: str) -> list[str]:
    segment = segment.replace("\\", "/")
    try:
        return shlex.split(segment)
    except ValueError:  # unbalanced quotes, e.g. a heredoc body line
        return segment.split()


def _writes_in_place(tokens: list[str]) -> bool:
    verb = tokens[0].lower()
    if verb == "git":
        return len(tokens) > 1 and tokens[1] in ("rm", "mv")
    if verb in ("sed", "perl"):  # only in-place edits write
        return any(re.match(r"^(?:-\w*i|--in-place)", t) for t in tokens[1:])
    return bool(MUTATING_VERB.match(verb))


def write_targets(command: str, cwd: Path) -> list[Path]:
    """Paths a shell command may write, as best a heuristic can tell."""
    targets: list[Path] = []
    for segment in SEGMENT_SPLIT.split(command):
        tokens = _tokens(segment.strip())
        if not tokens:
            continue
        if tokens[0] == "cd" and len(tokens) > 1:
            cwd = cwd / tokens[1]
            continue
        named: list[str] = []
        for i, token in enumerate(tokens):
            redirect = REDIRECT.match(token)
            if redirect:
                target = redirect.group(1) or (tokens[i + 1] if i + 1 < len(tokens) else "")
                if target and not target.startswith("&"):
                    named.append(target)
        if tokens[0] == "tee" or _writes_in_place(tokens):
            named.extend(tokens[1:])
        targets.extend(cwd / t for t in named)
    return targets


def main():
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        # Can't parse input, don't block
        return 0

    tool_input = data.get("tool_input", {})
    file_path = tool_input.get("file_path", "") or tool_input.get("path", "")
    command = tool_input.get("command", "")
    cwd = Path(data.get("cwd") or ".")

    candidates = [Path(file_path)] if file_path else write_targets(command, cwd)
    for path in candidates:
        if is_protected(path):
            # Ask rather than refuse: the user sees the exact edit and decides.
            print(json.dumps({"hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "ask",
                "permissionDecisionReason": (
                    f"Changes committed test file '{path}'. Approve only if the "
                    "test itself is wrong (the spec or data changed), not to "
                    "make a failing test pass."
                ),
            }}))
            return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
