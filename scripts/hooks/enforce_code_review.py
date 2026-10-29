"""
Enforce Code Review Hook (Stop)
================================
Blocks Claude from finishing a turn if the implement skill ran but
code-review was not invoked afterward.

Uses marker files to track skill invocations:
- .claude/.implement-ran  — written by the implement skill
- .claude/.review-ran     — written by the code-review skill

Blocks at most once in a row: when Claude Code reports stop_hook_active
(the turn is already continuing because a Stop hook blocked it), the stop
is allowed. Otherwise a marker Claude can't write loops the turn forever.
The implement marker is kept, so the next turn is reminded again.

Exit codes:
    0 — all clear (no implement ran, code-review was run, or already blocked once)
    2 — implement ran without code-review, block the stop (Claude Code only
        blocks on 2; any other non-zero code is a non-blocking warning)
"""

import json
import sys
from pathlib import Path

MARKERS_DIR = Path(".claude")
IMPLEMENT_MARKER = MARKERS_DIR / ".implement-ran"
REVIEW_MARKER = MARKERS_DIR / ".review-ran"


def _stop_hook_active() -> bool:
    """Whether Claude Code says this turn is already continuing from a Stop hook."""
    try:
        return bool(json.load(sys.stdin).get("stop_hook_active"))
    except (json.JSONDecodeError, AttributeError):
        return False


def main():
    # If implement didn't run, nothing to enforce
    if not IMPLEMENT_MARKER.exists():
        return 0

    # If code-review ran after implement, all good
    if REVIEW_MARKER.exists():
        # Clean up both markers
        IMPLEMENT_MARKER.unlink(missing_ok=True)
        REVIEW_MARKER.unlink(missing_ok=True)
        return 0

    # Already blocked once this turn — don't loop
    if _stop_hook_active():
        return 0

    # Implement ran but code-review didn't — block
    print(
        "BLOCKED: /implement ran but /code-review was not invoked. "
        "Run /code-review before finishing.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
