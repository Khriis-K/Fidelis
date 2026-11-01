"""Download a public data file once, and prove later runs have the same bytes.

USGS publishes no checksums for the Mars 2020 TRN mosaics, so we pin our own: the
first download prints its SHA-256, that goes into config/jezero.yaml, and every later
download (on any machine) must match it. That catches truncated transfers and silent
upstream changes, though it trusts whatever the first download got.

    uv run fidelis-fetch URL DEST [--sha256 HEX] [--record PATH]
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

CHUNK = 1 << 20
# Seconds a stalled connection may sit idle (per read, not for the whole download).
TIMEOUT_S = 60


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, dest: Path, sha256: str | None) -> str:
    """Download url to dest unless a valid copy is already there; return dest's SHA-256.

    With no pin, an existing file is kept as is: there is nothing to check it against,
    and re-downloading gigabytes on every run would be worse.
    """
    dest = Path(dest)
    if dest.exists():
        actual = sha256_of(dest)
        if sha256 is None or actual == sha256:
            return actual
        print(f"{dest} has SHA-256 {actual}, expected {sha256}; downloading again", file=sys.stderr)

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response, open(part, "wb") as f:
            while chunk := response.read(CHUNK):
                digest.update(chunk)
                f.write(chunk)
        actual = digest.hexdigest()
        if sha256 is not None and actual != sha256:
            raise ValueError(f"{url} downloaded with SHA-256 {actual}, but {sha256} is pinned")
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    # Rename, not write in place: an interrupted download never looks like a finished file.
    part.replace(dest)
    return actual


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url")
    parser.add_argument("dest", type=Path)
    parser.add_argument("--sha256", help="pinned hash; omit on the very first download")
    parser.add_argument("--record", type=Path, help="write the verified hash here (the workflow's output)")
    args = parser.parse_args()

    actual = fetch(args.url, args.dest, args.sha256)
    if args.sha256 is None:
        print(f"NOT PINNED: {args.dest} has SHA-256 {actual}; pin it in config/jezero.yaml", file=sys.stderr)
    if args.record is not None:
        tmp = args.record.with_name(args.record.name + ".tmp")
        # An unpinned run checked nothing; say so where the workflow keeps its proof.
        status = "verified" if args.sha256 else "UNPINNED"
        tmp.write_text(f"{actual}  {args.dest.name}  {status}\n")
        tmp.replace(args.record)
    print(f"{args.dest}: SHA-256 {actual}")
    return 0
