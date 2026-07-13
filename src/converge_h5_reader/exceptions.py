"""Exception hierarchy for :mod:`converge_h5_reader`."""

from __future__ import annotations

__all__ = [
    "ClosedFileError",
    "ConvergeH5Error",
    "MissingBackendError",
    "MissingFieldError",
    "StreamNotFoundError",
]


class ConvergeH5Error(Exception):
    """Base class for every error raised by this package."""


class StreamNotFoundError(ConvergeH5Error, KeyError):
    """The requested stream does not exist in the file."""


class MissingFieldError(ConvergeH5Error, KeyError):
    """The requested cell variable does not exist in the stream."""


class ClosedFileError(ConvergeH5Error, RuntimeError):
    """The file was closed and can no longer be accessed."""


class MissingBackendError(ConvergeH5Error, ImportError):
    """A mesh backend was requested but its third-party dependencies are absent."""
