"""Device pinning + OOM helpers (the CPU-default compute posture)."""

import os

import pytest

from harness import gpu


@pytest.fixture(autouse=True)
def _clean_device_env(monkeypatch):
    for var in ("JAX_PLATFORMS", "XLA_PYTHON_CLIENT_PREALLOCATE",
                "XLA_PYTHON_CLIENT_ALLOCATOR"):
        monkeypatch.delenv(var, raising=False)


def test_cpu_is_the_default_device():
    gpu.configure()
    assert os.environ["JAX_PLATFORMS"] == "cpu"
    # VRAM-hygiene variables are GPU-only; CPU selection must not set them.
    assert "XLA_PYTHON_CLIENT_PREALLOCATE" not in os.environ
    assert "XLA_PYTHON_CLIENT_ALLOCATOR" not in os.environ


def test_explicit_environment_wins():
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setenv("JAX_PLATFORMS", "cuda")
        gpu.configure()
        assert os.environ["JAX_PLATFORMS"] == "cuda"
    finally:
        monkey.undo()


def test_cuda_device_applies_vram_hygiene():
    gpu.configure(device="cuda")
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] == "platform"


def test_is_oom_walks_the_exception_chain():
    deep = RuntimeError("CUDA_ERROR_OUT_OF_MEMORY while allocating 2.1GiB")
    mid = RuntimeError("sweep failed")
    mid.__cause__ = deep
    assert gpu.is_oom(mid)
    assert not gpu.is_oom(ValueError("plain failure"))


def test_is_oom_no_infinite_loop_on_self_reference():
    exc = RuntimeError("loop")
    exc.__context__ = exc
    assert not gpu.is_oom(exc)
