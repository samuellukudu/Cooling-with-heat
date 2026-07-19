"""Advection operators using first-order upwind finite differences."""
import jax.numpy as jnp


def advection_1d(T: jnp.ndarray, u: jnp.ndarray, dx: float | jnp.ndarray) -> jnp.ndarray:
    """Compute -u * dT/dx using first-order upwind finite differences.

    Uses backward difference where u > 0 (flow left-to-right),
    forward difference where u <= 0 (flow right-to-left).

    Args:
        T: (N,) temperature field at cell centers.
        u: (N,) velocity field at cell centers.
        dx: Cell width. Scalar or (N,) array.

    Returns:
        (N,) advection term -u * dT/dx at cell centers.
    """
    T_forward = jnp.roll(T, -1)   # T[i+1]
    T_backward = jnp.roll(T, 1)   # T[i-1]

    forward_diff = (T_forward - T) / dx
    backward_diff = (T - T_backward) / dx

    # Upwind: use backward diff where u > 0, forward diff elsewhere
    return -u * jnp.where(u > 0, backward_diff, forward_diff)
