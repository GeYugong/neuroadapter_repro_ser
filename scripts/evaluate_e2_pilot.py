#!/usr/bin/env python3
"""Compatibility wrapper for the unified E2 evaluator."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_e2 import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
