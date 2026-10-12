import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from green import params  # noqa: E402


@pytest.fixture(autouse=True)
def easy_target(monkeypatch):
    # Every ticket wins, so tests do not have to grind. Tests that need real
    # difficulty set their own target.
    monkeypatch.setattr(params, "INITIAL_TARGET", params.MAX_TARGET)
