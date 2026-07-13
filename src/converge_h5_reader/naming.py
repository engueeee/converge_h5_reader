"""Filename and stream-name conventions of CONVERGE ``post*.h5`` output."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

__all__ = ["H5_PATTERN", "STREAM_PATTERN", "cad_from_filename", "normalize_stream"]

#: ``post000061_-7.40757e+01.h5`` -> the run crank angle in the capture group.
H5_PATTERN = re.compile(r"post\d+_([+-]?\d+\.?\d*(?:e[+-]?\d+)?)\.h5", re.IGNORECASE)

STREAM_PATTERN = re.compile(r"^STREAM_(\d+)$")


def normalize_stream(stream: int | str) -> str:
    """Return the canonical ``STREAM_NN`` group name.

    Accepts an index (``0``), a numeric string (``"1"``) or a full name
    (``"STREAM_00"``, case-insensitive, matching ``BOUNDARIES/STREAMS`` values
    such as ``b"Stream_00"``).

    Unlike the legacy ``converge_h5_to_df``, an unrecognised name is an error
    rather than a silent fall back to ``STREAM_00``.
    """
    if isinstance(stream, (int, np.integer)) and not isinstance(stream, bool):
        index = int(stream)
        if index < 0:
            raise ValueError(f"stream index must be >= 0, got {index}")
        return f"STREAM_{index:02d}"

    text = str(stream).strip()
    if text.isdigit():
        return f"STREAM_{int(text):02d}"

    match = STREAM_PATTERN.match(text.upper())
    if match is None:
        raise ValueError(
            f"cannot interpret {stream!r} as a stream; "
            'expected an index (0), a digit string ("1") or "STREAM_00"'
        )
    return f"STREAM_{int(match.group(1)):02d}"


def cad_from_filename(path: str | Path, *, strict: bool = True) -> float | None:
    """Extract the run crank angle encoded in a ``post*.h5`` filename.

    With ``strict=False`` a non-matching name yields ``None`` instead of raising.
    """
    name = Path(path).name
    match = H5_PATTERN.search(name)
    if match is None:
        if strict:
            raise ValueError(f"{name!r} does not match the CONVERGE post*.h5 naming convention")
        return None
    return float(match.group(1))
