#!/usr/bin/env python3
"""Example: Aspect Ratio Optimization of 2D Domains.

Demonstrates the relationship between domain geometry and the first eigenvalue
of the Laplacian, connecting to the Faber-Kahn and Rayleigh Quotient concepts
presented in the MIT 18.303 lecture notes.

For a rectangular domain of fixed area A = 1.0:
    Lx * Ly = 1.0  => Lx = sqrt(r), Ly = 1/sqrt(r)
where r is the aspect ratio Lx / Ly.

The smallest eigenvalue of the Laplacian with Dirichlet boundary conditions is:
    lambda_1 = pi^2 * (1/Lx^2 + 1/Ly^2) = pi^2 * (1/r + r)

The minimum is achieved at r = 1.0 (the square domain), where:
    lambda_1 = 2 * pi^2 approx 19.74

We use JAX's automatic differentiation to calculate the gradient of the computed
eigenvalue with respect to the aspect ratio r, and run gradient descent to
automatically optimize the domain's aspect ratio to find the square!

Run:
    python examples/05-eigenvalue-comparison/eigenvalue_comparison.py
"""
import jax
import jax.numpy as jnp

from diffheat import Grid2D, find_first_eigenvalue_2d


def compute_eigenvalue(r: float) -> float:
    """Compute the first eigenvalue for a 2D domain of area 1.0 and aspect ratio r."""
    # Area = Lx * Ly = 1.0
    # r = Lx / Ly => Lx = sqrt(r), Ly = 1 / sqrt(r)
    Lx = jnp.sqrt(r)
    Ly = 1.0 / jnp.sqrt(r)

    # Grid cell counts
    nx, ny = 24, 24
    grid = Grid2D.uniform(Lx=Lx, Ly=Ly, nx=nx, ny=ny)

    # Use inverse power iteration to find the smallest eigenvalue
    val, _ = find_first_eigenvalue_2d(grid, max_iter=12)
    return val


def main():
    print("=" * 60)
    print("DEMO: Aspect Ratio Optimization via Differentiable Eigenvalues")
    print("=" * 60)
    print("Analytical minimum is at r = 1.0 (Square), lambda_1 = 19.74")
    print("-" * 60)

    # 1. Sweep aspect ratios and print values
    print("Aspect Ratio Sweep:")
    r_sweep = [0.25, 0.5, 1.0, 2.0, 4.0]
    for r in r_sweep:
        val = compute_eigenvalue(r)
        analytic = (jnp.pi**2) * (1.0 / r + r)
        print(
            f"  r = {r:5.2f} | Lx = {jnp.sqrt(r):.3f}, Ly = {1.0/jnp.sqrt(r):.3f} "
            f"| Computed lambda_1 = {val:6.2f} (Analytic = {analytic:6.2f})"
        )

    # 2. Run Gradient-Based Optimization of aspect ratio r
    print("\nOptimizing Aspect Ratio r using JAX gradients...")
    r_init = 4.0  # Start with an elongated domain (aspect ratio 4:1)
    r = r_init
    lr = 0.15  # Learning rate

    # Gradient function
    grad_fn = jax.grad(compute_eigenvalue)

    print(f"Start: r = {r:.4f}")
    for step in range(16):
        val = compute_eigenvalue(r)
        grad = grad_fn(r)
        r_next = r - lr * grad
        # Project to keep r > 0.1
        r_next = jnp.maximum(r_next, 0.1)

        print(
            f"  Step {step:2d} | r = {float(r):.4f} | lambda_1 = {float(val):6.2f} "
            f"| d(lambda_1)/dr = {float(grad):+6.2f}"
        )
        r = r_next

    print(f"Optimal aspect ratio found: r = {float(r):.4f} (Square = 1.0)")
    print("=" * 60)

    # 3. Plot curve and save if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        print("\nMatplotlib found. Plotting aspect ratio sweep curve...")
        rs = jnp.linspace(0.2, 5.0, 50)
        vals = [float(compute_eigenvalue(float(ri))) for ri in rs]
        analytics = [(jnp.pi**2) * (1.0 / float(ri) + float(ri)) for ri in rs]

        plt.figure(figsize=(8, 5))
        plt.plot(rs, vals, "o-", label="Computed (diffheat)", color="#1f77b4")
        plt.plot(rs, analytics, "--", label="Analytical", color="#ff7f0e")
        plt.axvline(
            1.0, color="gray", linestyle=":", label="Optimal Shape (Square)"
        )
        plt.xlabel("Aspect Ratio (Lx / Ly)")
        plt.ylabel(r"First Eigenvalue $\lambda_1$")
        plt.title("First Eigenvalue of 2D Laplacian vs Aspect Ratio (Fixed Area = 1.0)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        import os

        os.makedirs("examples/05-eigenvalue-comparison", exist_ok=True)
        plt.savefig(
            "examples/05-eigenvalue-comparison/eigenvalue_aspect_ratio.png",
            dpi=150,
        )
        print(
            "Saved plot to examples/05-eigenvalue-comparison/eigenvalue_aspect_ratio.png"
        )
    except ImportError:
        print("\nMatplotlib not installed. Skipping plot generation.")


if __name__ == "__main__":
    main()
