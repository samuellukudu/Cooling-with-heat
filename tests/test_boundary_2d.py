# tests/test_boundary_2d.py
"""Tests for 2D boundary conditions."""
import pytest
import jax.numpy as jnp
from diffheat.mesh import Grid2D, BoundaryCondition2D
from diffheat.mesh.boundary import apply_boundary_conditions_2d
from diffheat.operators import laplacian_2d


class TestBoundaryCondition2D:
    def test_dirichlet_creation(self):
        bc = BoundaryCondition2D(
            left={"kind": "dirichlet", "value": 1.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "neumann", "value": 0.0},
            top={"kind": "neumann", "value": 0.0},
        )
        assert bc.left["kind"] == "dirichlet"
        assert bc.left["value"] == 1.0
        assert bc.right["kind"] == "dirichlet"
        assert bc.bottom["kind"] == "neumann"

    def test_raises_on_unknown_kind(self):
        with pytest.raises(ValueError):
            BoundaryCondition2D(
                left={"kind": "periodic", "value": 0.0},
                right={"kind": "dirichlet", "value": 0.0},
                bottom={"kind": "neumann", "value": 0.0},
                top={"kind": "neumann", "value": 0.0},
            )

    def test_frozen_dataclass(self):
        bc = BoundaryCondition2D(
            left={"kind": "dirichlet", "value": 1.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "neumann", "value": 0.0},
            top={"kind": "neumann", "value": 0.0},
        )
        with pytest.raises(Exception):
            bc.left = {"kind": "neumann", "value": 0.0}


class TestApplyBoundaryConditions2D:
    def test_returns_correct_shapes(self):
        nx, ny = 20, 20
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=nx, ny=ny)
        bc = BoundaryCondition2D(
            left={"kind": "dirichlet", "value": 1.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "dirichlet", "value": 0.5},
            top={"kind": "dirichlet", "value": 0.5},
        )
        T = jnp.zeros((nx, ny))
        L_T, b_source = apply_boundary_conditions_2d(
            lambda x: laplacian_2d(x, grid), grid, bc, T
        )
        assert L_T.shape == (nx, ny)
        assert b_source.shape == (nx, ny)

    def test_homogeneous_neumann_all_sides_has_zero_source(self):
        nx, ny = 15, 15
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=nx, ny=ny)
        bc = BoundaryCondition2D(
            left={"kind": "neumann", "value": 0.0},
            right={"kind": "neumann", "value": 0.0},
            bottom={"kind": "neumann", "value": 0.0},
            top={"kind": "neumann", "value": 0.0},
        )
        T = jnp.ones((nx, ny))
        L_T, b_source = apply_boundary_conditions_2d(
            lambda x: laplacian_2d(x, grid), grid, bc, T
        )
        assert jnp.allclose(b_source, 0.0)

    def test_dirichlet_boundary_source_is_nonzero(self):
        nx, ny = 10, 10
        grid = Grid2D.uniform(Lx=1.0, Ly=1.0, nx=nx, ny=ny)
        bc = BoundaryCondition2D(
            left={"kind": "dirichlet", "value": 100.0},
            right={"kind": "dirichlet", "value": 0.0},
            bottom={"kind": "dirichlet", "value": 50.0},
            top={"kind": "dirichlet", "value": 50.0},
        )
        T = jnp.zeros((nx, ny))
        L_T, b_source = apply_boundary_conditions_2d(
            lambda x: laplacian_2d(x, grid), grid, bc, T
        )
        assert not jnp.isclose(jnp.sum(jnp.abs(b_source)), 0.0)
