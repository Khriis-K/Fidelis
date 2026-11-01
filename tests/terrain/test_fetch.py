import hashlib

import pytest

from fidelis.terrain.fetch import fetch, sha256_of

CONTENT = b"pretend GeoTIFF bytes" * 1000


@pytest.fixture
def source(tmp_path):
    """A local file standing in for the USGS server; file:// URLs go through the same urllib path."""
    path = tmp_path / "server" / "mosaic.tif"
    path.parent.mkdir()
    path.write_bytes(CONTENT)
    return path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_sha256_of_matches_hashlib(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(CONTENT)

    assert sha256_of(path) == digest(CONTENT)


def test_fetch_downloads_and_returns_the_hash(source, tmp_path):
    dest = tmp_path / "raw" / "mosaic.tif"

    result = fetch(source.as_uri(), dest, sha256=digest(CONTENT))

    assert dest.read_bytes() == CONTENT
    assert result == digest(CONTENT)


def test_fetch_skips_a_file_that_is_already_present_and_valid(tmp_path):
    # The URL points nowhere, so any download attempt would raise.
    dest = tmp_path / "mosaic.tif"
    dest.write_bytes(CONTENT)
    missing = (tmp_path / "nothing-here.tif").as_uri()

    assert fetch(missing, dest, sha256=digest(CONTENT)) == digest(CONTENT)


def test_fetch_replaces_a_corrupt_file(source, tmp_path):
    # A truncated earlier download must not be mistaken for the real file.
    dest = tmp_path / "mosaic.tif"
    dest.write_bytes(CONTENT[:100])

    fetch(source.as_uri(), dest, sha256=digest(CONTENT))

    assert dest.read_bytes() == CONTENT


def test_fetch_rejects_a_download_that_does_not_match_the_pin(source, tmp_path):
    # Upstream silently changed the file, or the transfer was corrupted.
    dest = tmp_path / "mosaic.tif"

    with pytest.raises(ValueError, match="SHA-256"):
        fetch(source.as_uri(), dest, sha256=digest(b"what we pinned last time"))

    assert list(tmp_path.glob("mosaic.tif*")) == [], "a rejected download must leave nothing behind"


def test_fetch_without_a_pin_downloads_and_reports_the_hash_to_pin(source, tmp_path):
    # USGS publishes no checksums, so the first download is what gets pinned.
    dest = tmp_path / "mosaic.tif"

    assert fetch(source.as_uri(), dest, sha256=None) == digest(CONTENT)
    assert dest.read_bytes() == CONTENT
