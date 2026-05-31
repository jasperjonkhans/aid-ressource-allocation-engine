"""Compatibility package for notebooks started from project/notebooks/."""

from __future__ import annotations

from pathlib import Path


__path__ = [str(Path(__file__).resolve().parents[2])]
