"""Make absolute project imports work when Python starts inside project/."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_PARENT = Path(__file__).resolve().parent.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))
