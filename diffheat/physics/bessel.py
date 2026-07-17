# diffheat/physics/bessel.py
"""Bessel function eigenvalues and eigenfunctions for the disc.

Implements the analytical solutions from Hancock §8, §10, §14.2, §15.

On the unit disc D = {(x,y) : x² + y² ≤ 1}, the Sturm-Liouville
eigenvalues and eigenfunctions of the Laplacian with zero-Dirichlet BCs
are:

    λ_{m,n} = j_{m,n}²
    v_{m,n}(r,θ) = J_m(r · j_{m,n}) · {cos(mθ), sin(mθ)}

where j_{m,n} is the n-th positive zero of the Bessel function J_m.
"""
import jax.numpy as jnp
from scipy.special import jn_zeros, jv


def bessel_j_zero(m: int, n: int) -> float:
    """Return the *n*-th positive zero ``j_{m,n}`` of the Bessel function J_m.

    Uses :func:`scipy.special.jn_zeros`.

    Args:
        m: Order of the Bessel function (m = 0, 1, 2, …).
        n: Index of the zero (1-based, n = 1, 2, 3, …).

    Returns:
        The zero ``j_{m,n}``.

    Examples:
        >>> bessel_j_zero(0, 1)
        2.4048...   # j_{0,1}
        >>> bessel_j_zero(1, 1)
        3.8317...   # j_{1,1}
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if m < 0:
        raise ValueError(f"m must be >= 0, got {m}")
    zeros = jn_zeros(m, n)
    return float(zeros[n - 1])


def eigenvalue_disc(m: int, n: int, R: float = 1.0) -> float:
    """Eigenvalue λ_{m,n} of the Laplacian on a disc of radius *R*.

    λ_{m,n} = (j_{m,n} / R)²

    Args:
        m: Angular mode number.
        n: Radial mode number (1-based).
        R: Disc radius (default 1).

    Returns:
        Eigenvalue λ_{m,n}.
    """
    j = bessel_j_zero(m, n)
    return (j / R) ** 2


def eigenfunction_disc(
    r: jnp.ndarray,
    theta: jnp.ndarray,
    m: int,
    n: int,
    kind: str = "cos",
    R: float = 1.0,
) -> jnp.ndarray:
    """Evaluate the (m,n) eigenfunction of the Laplacian on a disc.

    v_{m,n}(r, θ) = J_m(r · j_{m,n} / R) · {cos(mθ)  if kind='cos'
                                         {sin(mθ)  if kind='sin'

    For m=0, the 'sin' branch is identically zero; use 'cos'.

    Args:
        r: Radial coordinate(s).  Scalar or array, 0 ≤ r ≤ R.
        theta: Angular coordinate(s).  Same shape as *r*.
        m: Angular mode number.
        n: Radial mode number (1-based).
        kind: ``'cos'`` or ``'sin'`` — angular factor.
        R: Disc radius (default 1).

    Returns:
        Eigenfunction values, same shape as *r*.
    """
    if m < 0:
        raise ValueError(f"m must be >= 0, got {m}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if kind not in ("cos", "sin"):
        raise ValueError(f"kind must be 'cos' or 'sin', got {kind!r}")

    j_mn = bessel_j_zero(m, n)
    scaled_r = jnp.asarray(r) * j_mn / R

    # Evaluate J_m on the scaled radial coordinate.
    # scipy.special.jv works on numpy arrays; convert as needed.
    radial = jnp.asarray(jv(m, jnp.asarray(scaled_r)))

    if m == 0:
        if kind == "sin":
            return jnp.zeros_like(jnp.asarray(r))
        angular = jnp.ones_like(jnp.asarray(theta))
    elif kind == "cos":
        angular = jnp.cos(m * jnp.asarray(theta))
    else:
        angular = jnp.sin(m * jnp.asarray(theta))

    return radial * angular


def eigenfunction_norm(m: int, n: int, R: float = 1.0) -> float:
    r"""L² norm of the (m,n) eigenfunction on the disc of radius *R*.

    ||v_{m,n}||² = ∫_D v_{m,n}² dA = π·R²·(1 + δ_{m,0})/2 · [J_{m+1}(j_{m,n})]²

    where δ_{m,0} is 1 if m=0 and 0 otherwise (Kronecker delta).

    This is used for computing expansion coefficients in spectral series.

    Args:
        m: Angular mode number.
        n: Radial mode number (1-based).
        R: Disc radius (default 1).

    Returns:
        Squared L² norm ||v_{m,n}||².
    """
    j_mn = bessel_j_zero(m, n)
    # J_{m+1}(j_{m,n})
    j_next = float(jv(m + 1, j_mn))

    factor = 2.0 if m == 0 else 1.0
    return 0.5 * factor * jnp.pi * R ** 2 * j_next ** 2
