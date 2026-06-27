# tests/test_boundary_3d.py
"""Tests for 3D boundary conditions."""
import jax.numpy as jnp
import pytest
from diffheat.mesh import Grid3D, BoundaryCondition3D
from diffheat.mesh.boundary import apply_boundary_conditions_3d
from diffheat.operators.laplacian import laplacian_3d


def _all_dirichlet(value: float = 0.0) -> BoundaryCondition3D:
    d = {"kind": "dirichlet", "value": value}
    return BoundaryCondition3D(
        left=d, right=d, bottom=d, top=d, front=d, back=d
    )


def _all_neumann(value: float = 0.0) -> BoundaryCondition3D:
    n = {"kind": "neumann", "value": value}
    return BoundaryCondition3D(
        left=n, right=n, bottom=n, top=n, front=n, back=n
    )


class TestBoundaryCondition3D:
    def test_dirichlet_creation(self):
        bc = _all_dirichlet(1.0)
        assert bc.left["kind"] == "dirichlet"
        assert bc.back["value"] == 1.0

    def test_neumann_creation(self):
        bc = _all_neumann(0.0)
        assert bc.front["kind"] == "neumann"

    def test_raises_on_unknown_kind(self):
        with pytest.raises(ValueError, match="Unknown boundary kind"):
            BoundaryCondition3D(
                left={"kind": "periodic", "value": 0.0},
                right={"kind": "dirichlet", "value": 0.0},
                bottom={"kind": "neumann", "value": 0.0},
                top={"kind": "neumann", "value": 0.0},
                front={"kind": "neumann", "value": 0.0},
                back={"kind": "neumann", "value": 0.0},
            )

    def test_frozen_dataclass(self):
        bc = _all_dirichlet()
        with pytest.raises(Exception):
            bc.left = {"kind": "neumann", "value": 0.0}  # type: ignore[misc]


class TestApplyBoundaryConditions3D:
    def test_returns_correct_shapes(self):
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=8, ny=10, nz=6)
        bc = _all_neumann()
        T = jnp.ones((8, 10, 6))
        L_T, b = apply_boundary_conditions_3d(
            lambda x: laplacian_3d(x, grid), grid, bc, T
        )
        assert L_T.shape == (8, 10, 6)
        assert b.shape == (8, 10, 6)

    def test_homogeneous_neumann_all_faces_has_zero_source(self):
        """Homogeneous Neumann (zero flux) should yield zero b_source."""
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=8, ny=8, nz=8)
        bc = _all_neumann(0.0)
        T = jnp.ones((8, 8, 8))
        _, b = apply_boundary_conditions_3d(
            lambda x: laplacian_3d(x, grid), grid, bc, T
        )
        assert jnp.allclose(b, 0.0)

    def test_dirichlet_boundary_source_is_nonzero(self):
        """Non-zero Dirichlet BCs should give non-zero b_source."""
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=8, ny=8, nz=8)
        bc = _all_dirichlet(1.0)
        T = jnp.zeros((8, 8, 8))
        _, b = apply_boundary_conditions_3d(
            lambda x: laplacian_3d(x, grid), grid, bc, T
        )
        assert not jnp.allclose(b, 0.0)

    def test_linear_field_matching_dirichlet_gives_zero_laplacian(self):
        """T = x: Laplacian is analytically zero; corrected stencil should be ~0."""
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=10, ny=10, nz=10)
        T = grid.X  # linear in x, so d^2T/dx^2 = d^2T/dy^2 = d^2T/dz^2 = 0
        bc = BoundaryCondition3D(
            left={"kind": "dirichlet", "value": float(grid.x_centers[0])},
            right={"kind": "dirichlet", "value": float(grid.x_centers[-1])},
            bottom={"kind": "neumann", "value": 0.0},
            top={"kind": "neumann", "value": 0.0},
            front={"kind": "neumann", "value": 0.0},
            back={"kind": "neumann", "value": 0.0},
        )
        L_T, b = apply_boundary_conditions_3d(
            lambda x: laplacian_3d(x, grid), grid, bc, T
        )
        total = L_T + b
        # Interior should be close to zero
        assert jnp.allclose(total[1:-1, 1:-1, 1:-1], 0.0, atol=1e-10)

    def test_constant_field_with_neumann_gives_zero_total(self):
        """Constant T with Neumann BCs: Laplacian is zero everywhere."""
        grid = Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=8, ny=8, nz=8)
        bc = _all_neumann(0.0)
        T = jnp.full((8, 8, 8), 5.0)
        L_T, b = apply_boundary_conditions_3d(
            lambda x: laplacian_3d(x, grid), grid, bc, T
        )
        assert jnp.allclose(L_T + b, 0.0, atol=1e-10)
