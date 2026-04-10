"""Unit tests for train_aic device resolution behavior."""

import pytest

from examples.train_aic import resolve_optimization_device


def test_resolve_device_auto_cuda_available():
    assert resolve_optimization_device("auto", cuda_available=True) == "cuda"


def test_resolve_device_auto_no_cuda():
    assert resolve_optimization_device("auto", cuda_available=False) == "cpu"


def test_resolve_device_cpu_overrides_to_cuda_when_available():
    assert resolve_optimization_device("cpu", cuda_available=True) == "cuda"


def test_resolve_device_cuda_without_cuda_raises():
    with pytest.raises(ValueError):
        resolve_optimization_device("cuda", cuda_available=False)


def test_resolve_device_keeps_cpu_without_cuda():
    assert resolve_optimization_device("cpu", cuda_available=False) == "cpu"
