import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modern_edge_toolkit.detectors_modern import SUPPORTED_METHODS


def test_supported_methods():
    assert len(SUPPORTED_METHODS) == 6
