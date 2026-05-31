"""Compatibility package for imports started from inside project/.

When the current working directory is the package directory itself, Python
looks for project/project before it can resolve absolute imports like
project.helper.predictions. This shim points the package search path back to
the real project package directory one level up.
"""

from __future__ import annotations

from pathlib import Path


__path__ = [str(Path(__file__).resolve().parents[1])]
