"""Checks C5 (raw tar) and C7 (interim sanity), and building the raw tar they need."""

import gzip
import hashlib
import io
import logging
import re
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import duckdb
import pyarrow.parquet as pq

from hsl_rt_analysis.hfp_archive.config import interim_day
from hsl_rt_analysis.hfp_archive.remote import Remote, RemoteEntry


logger = logging.getLogger(__name__)

INTERIM_VP_RE = re.compile(r"^(.+)_(\d{8})\.geoparquet$")
INTERIM_OTHER_RE = re.compile(r"^(.+)_(\d{8})_other_messages\.txt\.gz$")


# --- C5: the raw tar ----------------------------------------------------------------------------------


def build_tar(raw_dir: Path, names: list[str], tar_path: Path, day: str) -> Path:
    """Write the day's original files, unchanged and sorted, into an uncompressed pax tar.

    Owner fields are cleared so the tar depends only on the files; modification times are kept.

    Args:
        raw_dir: Local raw day directory.
        names: The day's file names.
        tar_path: Destination; written via a temporary name.
        day: The day, YYYY-MM-DD (the member directory).

    Returns:
        The tar path.
    """
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tar_path.with_name(f".{tar_path.name}.tmp")
    with tarfile.open(tmp, "w", format=tarfile.PAX_FORMAT) as tar:
        for name in sorted(names):
            path = raw_dir / name
            info = tar.gettarinfo(path, arcname=f"{day}/{name}")
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            with path.open("rb") as f:
                tar.addfile(info, f)
    tmp.replace(tar_path)
    return tar_path


def check_c5(tar_path: Path, expected: dict[str, str], day: str) -> dict[str, Any]:
    """Check that the tar holds exactly the day's files, each with its original SHA-256.

    Args:
        tar_path: The written tar, read back from disk.
        expected: Original file name mapped to the SHA-256 confirmed on the storage box.
        day: The day, YYYY-MM-DD.

    Returns:
        The check result.
    """
    seen: dict[str, str] = {}
    order: list[str] = []
    with tarfile.open(tar_path, "r:") as tar:
        for member in tar:
            f = tar.extractfile(member)
            if f is None or not member.name.startswith(f"{day}/"):
                seen[member.name] = "not a day file"
                continue
            digest = hashlib.sha256()
            while chunk := f.read(1 << 20):
                digest.update(chunk)
            name = member.name.removeprefix(f"{day}/")
            seen[name] = digest.hexdigest()
            order.append(name)
    mismatched = sorted(n for n in expected if seen.get(n) != expected[n])
    extra = sorted(set(seen) - set(expected))
    return {
        "members": len(seen),
        "mismatched": len(mismatched),
        "extra": len(extra),
        "sorted": order == sorted(order),
        "examples": (mismatched + extra)[:10],
        "ok": not mismatched and not extra and order == sorted(order),
    }


# --- C7: interim sanity -------------------------------------------------------------------------------


def route_group(source_file: str) -> str:
    """Return the interim route group of a raw file, as exploration/hfp_data_processor.py assigns it.

    Args:
        source_file: Raw file name, <route>-YYYY-MM-DDTHH.txt[.gz].

    Returns:
        The route itself for 1001-4999, "<digit>xxx" for 5000-9999, else "other".
    """
    route = source_file.split("-")[0][:4]
    if "1001" <= route <= "4999":
        return route
    if "5000" <= route <= "9999":
        return route[:1] + "xxx"
    return "other"


def parquet_rows_from_footer(remote: Remote, rel: str, size: int) -> int:
    """Read a remote Parquet file's row count from its footer only (a few KB instead of the whole file).

    Args:
        remote: The storage box.
        rel: File path relative to the archive root.
        size: File size in bytes.

    Returns:
        Number of rows.

    Raises:
        ValueError: If the file does not end like a Parquet file.
    """
    # One read of the last 64 KB holds the footer of nearly every file; a larger footer costs a second read.
    chunk = min(size, 1 << 16)
    tail = remote.read_range(rel, size - chunk, chunk)
    if tail[-4:] != b"PAR1":
        msg = f"{rel} is not a Parquet file"
        raise ValueError(msg)
    footer_len = int.from_bytes(tail[-8:-4], "little")
    if footer_len + 8 <= len(tail):
        footer = tail[-8 - footer_len : -8]
    else:
        footer = remote.read_range(rel, size - 8 - footer_len, footer_len)
    tail = tail[-8:]
    return pq.read_metadata(io.BytesIO(b"PAR1" + footer + tail)).num_rows


def count_lines(path: Path) -> int:
    """Count the non-empty lines of a gzip text file.

    Args:
        path: File.

    Returns:
        Number of non-empty lines.
    """
    with gzip.open(path, "rb") as f:
        return sum(1 for line in f if line.strip())


def archive_groups(file_stats: Path) -> dict[str, dict[str, int]]:
    """Sum the per-file raw line counts of the conversion by interim route group.

    Args:
        file_stats: file_stats.parquet written by the conversion.

    Returns:
        Group mapped to {"lines", "vp_lines"} (repeats included, as the interim builder saw them).
    """
    groups: dict[str, dict[str, int]] = defaultdict(lambda: {"lines": 0, "vp_lines": 0})
    rows = duckdb.sql(f"SELECT source_file, lines, vp_lines FROM read_parquet('{file_stats}')").fetchall()
    for source_file, lines, vp_lines in rows:
        g = groups[route_group(source_file)]
        g["lines"] += int(lines)
        g["vp_lines"] += int(vp_lines)
    return dict(groups)


def check_c7(remote: Remote, day: str, entries: list[RemoteEntry], file_stats: Path, scratch: Path) -> dict[str, Any]:
    """Check that no interim route group holds more data than the raw files it was built from.

    Per group: interim VP rows <= raw VP lines, and interim VP rows + other-message lines <= raw lines.
    Otherwise interim has data that raw lacks, and replacing interim would lose it.

    Args:
        remote: The storage box.
        day: The day, YYYY-MM-DD.
        entries: The interim day directory's entries.
        file_stats: file_stats.parquet of the conversion.
        scratch: Local directory for downloaded other-message files (emptied afterwards).

    Returns:
        The check result.
    """
    raw = archive_groups(file_stats)
    interim: dict[str, dict[str, int]] = defaultdict(lambda: {"vp_rows": 0, "other_lines": 0})
    unexpected = []
    stamp = day.replace("-", "")
    has_other = False
    for e in entries:
        vp, other = INTERIM_VP_RE.match(e.name), INTERIM_OTHER_RE.match(e.name)
        if other and other.group(2) == stamp:
            has_other = True
        elif vp and vp.group(2) == stamp and not e.is_dir:
            interim[vp.group(1)]["vp_rows"] += parquet_rows_from_footer(remote, f"{interim_day(day)}/{e.name}", e.size)
        else:
            unexpected.append(e.name)
    if has_other:
        remote.download_dir(interim_day(day), scratch, include="*_other_messages.txt.gz")
        for path in sorted(scratch.glob("*_other_messages.txt.gz")):
            match = INTERIM_OTHER_RE.match(path.name)
            if match:
                interim[match.group(1)]["other_lines"] += count_lines(path)
            path.unlink()
    over = []
    for group, got in sorted(interim.items()):
        have = raw.get(group, {"lines": 0, "vp_lines": 0})
        if got["vp_rows"] > have["vp_lines"] or got["vp_rows"] + got["other_lines"] > have["lines"]:
            over.append({"group": group, **got, "raw_lines": have["lines"], "raw_vp_lines": have["vp_lines"]})
    return {
        "groups": len(interim),
        "interim_vp_rows": sum(g["vp_rows"] for g in interim.values()),
        "interim_other_lines": sum(g["other_lines"] for g in interim.values()),
        "raw_vp_lines": sum(g["vp_lines"] for g in raw.values()),
        "groups_over": over[:20],
        "unexpected_files": unexpected[:20],
        "ok": not over and not unexpected,
    }
