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


def advection_2d(
    T: jnp.ndarray,
    u_x: jnp.ndarray,
    u_y: jnp.ndarray,
    dx: float | jnp.ndarray,
    dy: float | jnp.ndarray,
) -> jnp.ndarray:
    """Compute -(u_x * dT/dx + u_y * dT/dy) using first-order upwind.

    Upwinding is applied independently in x and y.
    Uses backward difference where velocity component > 0,
    forward difference where velocity component <= 0.

    Args:
        T: (nx, ny) temperature field at cell centers.
        u_x: (nx, ny) x-velocity field.
        u_y: (nx, ny) y-velocity field.
        dx: x cell width. Scalar or (nx,) array.
        dy: y cell width. Scalar or (ny,) array.

    Returns:
        (nx, ny) advection term at cell centers.
    """
    # --- x-direction upwind ---
    # Broadcast dx to (nx, 1) for dividing (nx, ny) arrays
    _dx = dx[:, jnp.newaxis] if dx.ndim == 1 else dx
    T_forward_x = jnp.roll(T, -1, axis=0)
    T_backward_x = jnp.roll(T, 1, axis=0)
    forward_diff_x = (T_forward_x - T) / _dx
    backward_diff_x = (T - T_backward_x) / _dx
    adv_x = -u_x * jnp.where(u_x > 0, backward_diff_x, forward_diff_x)

    # --- y-direction upwind ---
    # Broadcast dy to (1, ny) for dividing (nx, ny) arrays
    _dy = dy[jnp.newaxis, :] if dy.ndim == 1 else dy
    T_forward_y = jnp.roll(T, -1, axis=1)
    T_backward_y = jnp.roll(T, 1, axis=1)
    forward_diff_y = (T_forward_y - T) / _dy
    backward_diff_y = (T - T_backward_y) / _dy
    adv_y = -u_y * jnp.where(u_y > 0, backward_diff_y, forward_diff_y)

    return adv_x + adv_y


def advection_3d(
    T: jnp.ndarray,
    u_x: jnp.ndarray,
    u_y: jnp.ndarray,
    u_z: jnp.ndarray,
    dx: float | jnp.ndarray,
    dy: float | jnp.ndarray,
    dz: float | jnp.ndarray,
) -> jnp.ndarray:
    """Compute -(u_x*dT/dx + u_y*dT/dy + u_z*dT/dz) using first-order upwind.

    Upwinding is applied independently in x, y, and z.

    Args:
        T: (nx, ny, nz) temperature field at cell centers.
        u_x: (nx, ny, nz) x-velocity field.
        u_y: (nx, ny, nz) y-velocity field.
        u_z: (nx, ny, nz) z-velocity field.
        dx: x cell width. Scalar or (nx,) array.
        dy: y cell width. Scalar or (ny,) array.
        dz: z cell width. Scalar or (nz,) array.

    Returns:
        (nx, ny, nz) advection term at cell centers.
    """
    # --- x-direction upwind ---
    _dx = dx[:, jnp.newaxis, jnp.newaxis] if isinstance(dx, jnp.ndarray) and dx.ndim >= 1 else dx
    T_forward_x = jnp.roll(T, -1, axis=0)
    T_backward_x = jnp.roll(T, 1, axis=0)
    forward_diff_x = (T_forward_x - T) / _dx
    backward_diff_x = (T - T_backward_x) / _dx
    adv_x = -u_x * jnp.where(u_x > 0, backward_diff_x, forward_diff_x)

    # --- y-direction upwind ---
    _dy = dy[jnp.newaxis, :, jnp.newaxis] if isinstance(dy, jnp.ndarray) and dy.ndim >= 1 else dy
    T_forward_y = jnp.roll(T, -1, axis=1)
    T_backward_y = jnp.roll(T, 1, axis=1)
    forward_diff_y = (T_forward_y - T) / _dy
    backward_diff_y = (T - T_backward_y) / _dy
    adv_y = -u_y * jnp.where(u_y > 0, backward_diff_y, forward_diff_y)

    # --- z-direction upwind ---
    _dz = dz[jnp.newaxis, jnp.newaxis, :] if isinstance(dz, jnp.ndarray) and dz.ndim >= 1 else dz
    T_forward_z = jnp.roll(T, -1, axis=2)
    T_backward_z = jnp.roll(T, 1, axis=2)
    forward_diff_z = (T_forward_z - T) / _dz
    backward_diff_z = (T - T_backward_z) / _dz
    adv_z = -u_z * jnp.where(u_z > 0, backward_diff_z, forward_diff_z)

    return adv_x + adv_y + adv_z
