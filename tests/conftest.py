"""Shared pytest fixtures for cello sampler tests."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest


def pytest_configure(config: pytest.Config) -> None:  # noqa: ARG001
    """Install lightweight stubs for optional heavy dependencies.

    Allows the test suite to run without TensorFlow or resampy installed.
    """
    if "resampy" not in sys.modules:
        stub_resampy = MagicMock()
        stub_resampy.resample.return_value = np.zeros(160, dtype=np.float64)
        sys.modules["resampy"] = stub_resampy

    if "crepe" not in sys.modules:
        sys.modules["crepe"] = MagicMock()
