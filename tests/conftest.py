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
    # Where every ticket is a block no ticket can be a share, so the shares' part of
    # each block's emission would just stay in the reserve: the finder gets it all.
    # Tests of the share economy set SHARES_PERCENT (and a real target) themselves.
    monkeypatch.setattr(params, "SHARES_PERCENT", 0)
    # Tests pay new addresses a few yaprak at a time; the minimum for opening an
    # account is tested on its own (test_fees.py sets NEW_ACCOUNT_MIN).
    monkeypatch.setattr(params, "NEW_ACCOUNT_MIN", 0)
