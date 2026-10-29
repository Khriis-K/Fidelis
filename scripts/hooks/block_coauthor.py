"""PreToolUse hook: block Bash commands that attempt co-author attribution.

CLAUDE.md Git Rules: all commits must be authored solely by Christopher Kuizon.
This catches `git commit ... -m "... Co-Authored-By: ..."` before it runs.
"""
import json
import re
import sys

PATTERN = re.compile(r"co[-_ ]?authored[-_ ]?by\s*:", re.IGNORECASE)


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0  # can't parse; don't block
    command = data.get("tool_input", {}).get("command", "")
    if PATTERN.search(command):
        print(
            "Blocked: co-author trailer detected. Per CLAUDE.md Git Rules, "
            "commits must be authored solely by Christopher Kuizon.",
            file=sys.stderr,
        )
        return 2  # block the tool call
    return 0


if __name__ == "__main__":
    sys.exit(main())
