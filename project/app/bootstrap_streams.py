"""
Windowed-build stream guard.

In a PyInstaller --windowed / --noconsole build, the process has no console,
so sys.stdout and sys.stderr are None. Any library that writes to them — timm
and torch download progress bars, tqdm, logging to stderr, stray print() — will
raise:

    AttributeError: 'NoneType' object has no attribute 'write'

This module replaces missing streams with a safe sink so those writes succeed
silently. It MUST run before any heavy import (torch, timm, loguru) so that the
streams exist by the time those libraries touch them.

Usage — make this the FIRST thing imported in app/main.py:

    import app.bootstrap_streams  # noqa: F401  (must be first)
    # ... then the rest of the imports

It is a no-op in a normal console run, so it is safe in every environment.
"""

from __future__ import annotations

import io
import sys


class _NullWriter(io.TextIOBase):
    """A writable stream that discards everything. Implements the full
    text-stream interface libraries probe for (write, flush, isatty, etc.)."""

    def write(self, _s):  # noqa: D401
        return 0

    def flush(self):
        return None

    def isatty(self):
        return False

    def writable(self):
        return True

    def fileno(self):
        # Some libraries call fileno(); raise the conventional error so they
        # fall back gracefully rather than crashing.
        raise io.UnsupportedOperation("fileno")


def install() -> None:
    """Replace any None standard stream with a null sink. Idempotent."""
    if sys.stdout is None:
        sys.stdout = _NullWriter()
    if sys.stderr is None:
        sys.stderr = _NullWriter()
    if sys.__stdout__ is None:
        sys.__stdout__ = sys.stdout
    if sys.__stderr__ is None:
        sys.__stderr__ = sys.stderr


# Install on import so a bare `import app.bootstrap_streams` is enough.
install()
