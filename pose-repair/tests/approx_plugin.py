import numpy as np
import pytest


def pytest_configure(config):
    original = pytest.approx

    def array_aware(expected, *args, **kwargs):
        if isinstance(expected, list) and expected and isinstance(expected[0], list):
            expected = np.asarray(expected)
        return original(expected, *args, **kwargs)

    pytest.approx = array_aware
