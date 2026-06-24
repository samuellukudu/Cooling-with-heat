import pytest
import jax
import jax.numpy as jnp
from diffheat.utils import get_device, get_default_dtype, array


def test_get_device_returns_string():
    device = get_device()
    assert isinstance(device, str)
    assert device in ("cpu", "gpu", "tpu")


def test_get_device_matches_jax_backend():
    device = get_device()
    assert device == jax.default_backend()


def test_get_default_dtype_returns_jax_dtype():
    dtype = get_default_dtype()
    assert dtype in (jnp.float32, jnp.float64)


def test_get_default_dtype_is_float64_on_cpu():
    dtype = get_default_dtype()
    if get_device() == "cpu":
        assert dtype == jnp.float64
    else:
        assert dtype == jnp.float32


def test_array_creates_with_default_dtype():
    data = [1.0, 2.0, 3.0]
    result = array(data)
    assert result.dtype == get_default_dtype()
    assert result.shape == (3,)


def test_array_respects_explicit_dtype():
    data = [1.0, 2.0, 3.0]
    result = array(data, dtype=jnp.float16)
    assert result.dtype == jnp.float16


def test_array_preserves_values():
    data = [1.0, 2.0, 3.0]
    result = array(data)
    assert jnp.allclose(result, jnp.array([1.0, 2.0, 3.0]))
