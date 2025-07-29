"""Test package configuration.

Ensures project root is on `sys.path` so that imports like `import talamo` work
when running `pytest` from any location.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
