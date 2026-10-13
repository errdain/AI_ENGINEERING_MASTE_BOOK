import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "code"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


import pytest
import torch


@pytest.fixture(autouse=True)
def float64_default_for_this_lab():
    """Tier 3 lab comparisons use float64; restore the global default afterwards so other tiers' tests are unaffected."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)
