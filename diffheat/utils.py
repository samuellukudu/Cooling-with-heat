"""Device detection and array utilities for diffheat."""
import jax
import jax.numpy as jnp


def get_device() -> str:
    """Return the active JAX backend: 'cpu', 'gpu', or 'tpu'."""
    return jax.default_backend()


def get_default_dtype() -> jnp.dtype:
    """Return float32 on GPU/TPU, float64 on CPU."""
    if get_device() == "cpu":
        return jnp.float64
    return jnp.float32


def array(data, dtype=None) -> jnp.ndarray:
    """Create a JAX array on the default device with the default dtype.

    Args:
        data: Array-like data to convert.
        dtype: Optional explicit dtype. If None, uses get_default_dtype().

    Returns:
        jnp.ndarray on the active device.
    """
    if dtype is None:
        dtype = get_default_dtype()
    return jnp.array(data, dtype=dtype)
