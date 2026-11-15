"""Per-application pages for the frozen diffheat PDE zoo + absorption chiller.

Recover-and-run, not reimplementation: the PDE trial zoo was removed from
the tree at ``f36e929`` (DL-focus simplification) but its physics is frozen
in git history. This generator extracts that exact commit into
``data_cache/_frozen/`` (gitignored — never back into the tree), runs the
original solvers headless, and embeds downsampled trajectories in
standalone offline HTML pages under ``docs/apps/``:

```bash
python -m harness.viz_pde --out docs/apps            # all ten PDE pages
python -m harness.viz_pde --out /tmp/pde --quick     # fast small-grid pass
python -m harness.viz_pde --out docs/apps --only boussinesq,cloak
```

Pages (``docs/applications.md`` Part II is the design reference):

- ``pde-heat1d`` — slab transient + steady line + first-eigenvalue decay check
- ``pde-heat2d`` — hot/cold-edge plate approaching the linear steady state
- ``pde-heat3d`` — cube with three animated mid-plane slices
- ``pde-disc`` — heat on a disc from the *analytic* Bessel series (exact)
- ``pde-forced-conv`` — channel flow over a heated chip (Pe readout + glyphs)
- ``pde-boussinesq`` — Rayleigh–Bénard cavity at 3 Ra presets (plumes,
  velocity glyphs, resolved-vs-correlation Nu, validated-envelope flags)
- ``pde-cloak`` — thermal cloak at 3 ring presets (kappa minimap + objectives)
- ``pde-telegrapher`` — thermal pulse at 3 damping presets (finite-speed proof)
- ``pde-thermoelectric`` — lumped pair curves (live) + segmented-leg profile
- ``absorption`` — single-effect chiller machine (live tree core, not frozen)

Every page states its provenance (frozen SHA / live-tree note, grid, dt,
CFL); presets are separate embedded runs, never interpolations.
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

FROZEN_SHA = "42b97c755c37eb3fa301c47a91ef5f769e4e2f8c"
FROZEN_SHORT = FROZEN_SHA[:7]

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE = REPO_ROOT / "data_cache" / "_frozen" / FROZEN_SHORT

PDE_PAGES: tuple[tuple[str, str, str], ...] = (
    ("pde-heat1d", "Heat 1D — transient slab", "Dirichlet slab relaxing to its linear steady state."),
    ("pde-heat2d", "Heat 2D — hot/cold plate", "Square plate driven to the linear steady gradient."),
    ("pde-heat3d", "Heat 3D — cube slices", "Cube with three animated mid-plane slices."),
    ("pde-disc", "Heat on a disc — analytic", "Exact Bessel-series decay on a circular domain."),
    ("pde-forced-conv", "Forced convection — channel + chip", "Prescribed channel flow sweeping a heated chip."),
    ("pde-boussinesq", "Boussinesq cavity — convection onset", "Resolved Rayleigh–Bénard flow at three Ra presets."),
    ("pde-cloak", "Thermal cloak — ring design", "Ring-diffusivity cloak vs bare plate objectives."),
    ("pde-telegrapher", "Telegrapher — thermal pulse", "Finite-speed heat waves at three damping presets."),
    ("pde-thermoelectric", "Thermoelectric — pair + leg", "Lumped pair curves plus the resolved segmented leg."),
    ("pde-coldspot", "Hot-driven cold spot — boundary control", "Tracked avenue: a localized sink carves a cold spot in a hot-walled plate."),
    ("pde-junction", "Heat-to-work proxy — material junction", "Tracked avenue: steady drop across a conductivity jump vs uniform rod."),
    ("pde-metamaterial", "Metamaterial — ring inverse design", "Tracked avenue: first grid-search pass over ring parameters through the frozen solver."),
    ("absorption", "Absorption chiller — 4-vessel machine", "Single-effect LiBr/NH3 cycle (live tree core)."),
)

_BG, _FG, _AX, _AMB, _GRN, _CYN, _PNL, _SPN, _RED = (
    "#0f1115", "#e2e8f0", "#94a3b8", "#f59e0b", "#22c55e", "#06b6d4",
    "#161a21", "#1e222a", "#f87171")

JS_CMAP = """
function inferno(f) {
  f = Math.min(Math.max(f, 0), 1);
  const stops = [[13,13,31],[59,15,112],[140,29,64],[199,92,20],[249,200,70]];
  const x = f * (stops.length - 1), i = Math.min(Math.floor(x), stops.length - 2), t = x - i;
  const c = stops[i].map((v, k) => Math.round(v + t * (stops[i + 1][k] - v)));
  return "rgb(" + c[0] + "," + c[1] + "," + c[2] + ")";
}
"""

JS_FIELD = """
function paintGrid(p, frame, lo, hi) {
  const F = frame;
  for (let i = 0; i < F.length; i++) for (let j = 0; j < F[i].length; j++) {
    const el = document.getElementById(p + "-" + i + "-" + j);
    if (el) el.setAttribute("fill", inferno((F[i][j] - lo) / Math.max(hi - lo, 1e-9)));
  }
}
function wireScrub(sliderId, btnId, n, draw) {
  const sl = document.getElementById(sliderId);
  sl.addEventListener("input", (e) => draw(parseInt(e.target.value, 10)));
  let playing = false, timer = null;
  document.getElementById(btnId).addEventListener("click", () => {
    playing = !playing; document.getElementById(btnId).textContent = playing ? "⏸ pause" : "▶ play";
    if (playing) timer = setInterval(() => {
      sl.value = (parseInt(sl.value, 10) + 1) % n; draw(parseInt(sl.value, 10));
    }, 140);
    else clearInterval(timer);
  });
}
function wirePresets(runIds, draw) {
  runIds.forEach((id, k) => document.getElementById(id).addEventListener("click", () => {
    runIds.forEach((o, q) => document.getElementById(o).style.borderColor = q === k ? "#f59e0b" : "#1e222a");
    draw(k);
  }));
}
"""


# --------------------------------------------------------------------------
# Frozen-code management (extract once into gitignored data_cache; alias the
# frozen harness tree so it never shadows the live package).
# --------------------------------------------------------------------------

def frozen_pkg() -> Path:
    """Extract the frozen commit (once) and return the alias package dir.

    Layout under ``CACHE`` (gitignored): the verbatim ``diffheat``,
    ``harness`` and ``examples`` trees plus a ``frozenpkg/`` alias dir
    whose ``diffheat`` / ``frozen_harness`` symlinks let the frozen code
    import without ever shadowing the live ``harness`` package.
    """
    import shutil  # noqa: PLC0415

    pkg = CACHE / "frozenpkg"
    marker = CACHE / "MARKER"
    want = FROZEN_SHA + "\nno-top-init\n"
    if marker.exists() and marker.read_text() == want \
            and (pkg / "frozen_harness").is_symlink():
        return pkg
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="frozen_extract_"))
    try:
        with open(tmp / "fz.tar", "wb") as fh:
            subprocess.run(["git", "-C", str(REPO_ROOT), "archive", FROZEN_SHA,
                            "diffheat", "harness", "examples"],
                           check=True, stdout=fh)
        subprocess.run(["tar", "-x", "-C", str(tmp), "-f", str(tmp / "fz.tar")],
                       check=True)
        for name in ("diffheat", "harness", "examples"):
            dst = CACHE / name
            if dst.is_symlink():
                dst.unlink()
            elif dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(tmp / name), str(dst))
        pkg.mkdir(parents=True, exist_ok=True)
        for link, target in (("diffheat", "../diffheat"),
                             ("frozen_harness", "../harness")):
            lp = pkg / link
            if lp.is_symlink() or lp.exists():
                if lp.is_symlink():
                    lp.unlink()
                else:
                    shutil.rmtree(lp)
            lp.symlink_to(target, target_is_directory=True)
        # The frozen harness materials loader needs its anchor table path.
        ad = CACHE / "adsorbent-ml" / "data"
        ad.mkdir(parents=True, exist_ok=True)
        live_anchors = REPO_ROOT / "adsorbent-ml" / "data" / "anchors.csv"
        if not (ad / "anchors.csv").exists() and live_anchors.exists():
            (ad / "anchors.csv").symlink_to(live_anchors)
        # The frozen top-level __init__ wires backends/envs (gym registration
        # side effects); the PDE pages only ever need harness.physics.*,
        # so neutralize it in this disposable cache (the live package is
        # untouched — this tree is gitignored extraction output).
        (CACHE / "harness" / "__init__.py").write_text(
            '"""Frozen harness tree: PDE pages use harness.physics.* only."""\n')
        marker.write_text(want)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return pkg


def fmod(name: str):
    """Import ``frozen_harness.physics.<name>`` (cached by import machinery)."""
    import importlib  # noqa: PLC0415

    pkg = frozen_pkg()
    if str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))
    return importlib.import_module(f"frozen_harness.physics.{name}")


_frozen_mods: dict | None = None


def fro() -> dict:
    """Import the frozen packages (cached). Keys: dh, fhp, dhp, dhs, dhe."""
    global _frozen_mods
    if _frozen_mods is not None:
        return _frozen_mods
    pkg = frozen_pkg()
    if str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))
    import importlib  # noqa: PLC0415

    dh = importlib.import_module("diffheat")
    fhp = importlib.import_module("frozen_harness.physics.boussinesq")
    mods = {"dh": dh,
            "fhp": importlib.import_module("frozen_harness.physics"),
            "dhp": importlib.import_module("diffheat.physics"),
            "dhs": importlib.import_module("diffheat.solvers"),
            "dhe": importlib.import_module("diffheat.solvers.eigen")}
    _frozen_mods = mods
    return mods


# --------------------------------------------------------------------------
# Scenario runners (each returns JSON-serializable page data).
# --------------------------------------------------------------------------

def _thin_traj(traj, n_snap: int):
    import numpy as np  # noqa: PLC0415

    a = np.asarray(traj, dtype=float)
    idx = np.unique(np.linspace(0, len(a) - 1, min(n_snap, len(a))).astype(int))
    return a[idx], idx


def run_heat1d(*, quick: bool = False) -> dict:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)
    m = fro()
    N = 32 if quick else 64
    L, alpha = 1.0, 0.01
    grid = m["dh"].Grid1D.uniform(L, N)
    from diffheat.mesh.boundary import BoundaryCondition  # noqa: PLC0415

    bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))
    eqn = m["dhp"].HeatEquation1D(grid=grid, bc=bc, alpha=alpha)
    T0 = jnp.zeros(N)
    dt = 0.9 * float(jnp.min(grid.dx)) ** 2 / (2 * alpha)
    t_end = 8.0 if not quick else 4.0
    traj = m["dhs"].solve_heat_1d(eqn, T0, (0.0, t_end), dt)
    frames, idx = _thin_traj(traj, 21 if quick else 41)
    lam1 = float(m["dhe"].find_first_eigenvalue_1d(grid)[0])
    import math  # noqa: PLC0415

    xs = (np_centers(grid, N, L)).tolist()
    return {"xs": xs, "frames": [[round(float(v), 4) for v in row] for row in frames],
            "t": [round(float(i) * dt, 4) for i in idx],
            "steady": [round(1.0 - x, 4) for x in xs],
            "lambda1": lam1, "lambda1_exact": math.pi ** 2,
            "alpha": alpha, "L": L, "N": N, "dt": dt, "t_end": t_end}


def np_centers(grid, N: int, L: float) -> list[float]:
    import numpy as np  # noqa: PLC0415

    return (np.arange(N, dtype=float) + 0.5) * L / N


def run_heat2d(*, quick: bool = False) -> dict:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)
    m = fro()
    N = 24 if quick else 48
    grid = m["dh"].Grid2D.uniform(Lx=1.0, Ly=1.0, nx=N, ny=N)
    bc = m["dh"].BoundaryCondition2D(
        left={"kind": "dirichlet", "value": 100.0},
        right={"kind": "dirichlet", "value": 0.0},
        bottom={"kind": "neumann", "value": 0.0},
        top={"kind": "neumann", "value": 0.0})
    eqn = m["dhp"].HeatEquation2D(grid=grid, bc=bc, alpha=0.01)
    # t_end=30 ≈ 3 diffusion times (L²/π²α ≈ 10): the mean verifiably lands
    # on the linear steady state (50 °C) instead of a mid-transient.
    traj = m["dhs"].solve_heat_2d(eqn, jnp.zeros((N, N)), (0.0, 30.0), 0.0025)
    frames, idx = _thin_traj(traj, 21 if quick else 41)
    emb = frames[:, ::2, ::2] if N > 24 else frames
    return {"nx": int(emb.shape[1]), "ny": int(emb.shape[2]),
            "frames": [[[round(float(v), 2) for v in row] for row in fr] for fr in emb],
            "t": [round(float(i) * 0.0025, 4) for i in idx],
            "solved_n": N, "alpha": 0.01, "steady_mean": 50.0,
            "final_mean": round(float(frames[-1].mean()), 2)}


def run_heat3d(*, quick: bool = False) -> dict:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)
    m = fro()
    N = 12 if quick else 20
    grid = m["dh"].Grid3D.uniform(Lx=1.0, Ly=1.0, Lz=1.0, nx=N, ny=N, nz=N)
    bc = m["dh"].BoundaryCondition3D(
        left={"kind": "dirichlet", "value": 100.0},
        right={"kind": "dirichlet", "value": 0.0},
        bottom={"kind": "neumann", "value": 0.0},
        top={"kind": "neumann", "value": 0.0},
        front={"kind": "neumann", "value": 0.0},
        back={"kind": "neumann", "value": 0.0})
    eqn = m["dhp"].HeatEquation3D(grid=grid, bc=bc, alpha=0.01)
    traj = m["dhs"].solve_heat_3d(eqn, jnp.zeros((N, N, N)), (0.0, 2.0),
                                  0.002, save_every=20)
    frames, idx = _thin_traj(traj, 21 if quick else 31)
    mid = N // 2
    out = {}
    for name, sl in (("xy", frames[:, :, :, mid]), ("xz", frames[:, :, mid, :]),
                     ("yz", frames[:, mid, :, :])):
        out[name] = [[[round(float(v), 2) for v in row] for row in fr] for fr in sl]
    return {"N": N, "slices": out,
            "t": [round(float(i) * 0.002 * 20, 4) for i in idx]}


def run_disc(*, quick: bool = False) -> dict:
    """Analytic Bessel-series decay of an off-center bump (exact)."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)
    m = fro()
    R, sig = 1.0, 0.25
    u0 = lambda r, th: jnp.exp(-((r - 0.45) ** 2) / (2 * sig ** 2)) * (1.0 + 0.3 * jnp.cos(th))  # noqa: E731
    n_t = 21 if quick else 41
    traj, times, _grid = m["dh"].solve_heat_disc_analytical(
        u0, R, (0.0, 0.5), n_t, m_max=6 if quick else 8,
        n_max=6 if quick else 8, nr=40, ntheta=120)
    import numpy as np  # noqa: PLC0415

    full = np.asarray(traj, dtype=float)
    # Embed decimated polar grid (display resolution; solved finer).
    emb = full[:, ::2, ::3]
    peak = [round(float(full[i].max()), 5) for i in range(len(full))]
    return {"nr": int(emb.shape[1]), "nsec": int(emb.shape[2]), "R": R,
            "frames": [[[round(float(v), 4) for v in row] for row in fr] for fr in emb],
            "t": [round(float(t), 4) for t in np.asarray(times, dtype=float)],
            "peak": peak, "t_end": 0.5, "exact": True}


def run_forced(*, quick: bool = False) -> dict:
    """Channel flow over a heated chip (frozen forced-conv rollout)."""
    import jax  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)
    m = fro()
    fc = fmod("forced_conv")
    nx, ny = (48, 12) if quick else (80, 20)
    U, alpha = 1.0, fc.DEFAULT_ALPHA
    n_steps = 1000
    out = fc.simulate_forced_conv(
        inlet_velocity=U, nx=nx, ny=ny, dt=0.002, n_steps=n_steps,
        collect_trace=True)
    traj = out["trajectory"]
    frames, idx = _thin_traj(traj, 21 if quick else 41)
    emb = frames[:, ::2, ::2]
    import numpy as np  # noqa: PLC0415

    return {"nx": int(emb.shape[1]), "ny": int(emb.shape[2]),
            "frames": [[[round(float(v), 3) for v in row] for row in fr] for fr in emb],
            "t": [round(float(i) * 0.002, 4) for i in idx],
            "U": U, "alpha": alpha, "Pe": U * 4.0 / alpha,
            "chip_amp": float(fc.DEFAULT_SOURCE_AMP),
            "chip_x0": float(fc.DEFAULT_CHIP_X0),
            "chip_y0": float(fc.DEFAULT_CHIP_Y0),
            "Lx": 4.0, "Ly": 1.0,
            "tmax": round(float(np.asarray(traj).max()), 3)}


def run_boussinesq(*, quick: bool = False) -> dict:
    """RB cavity at Ra presets: T movies + steady velocity glyphs + Nu."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)
    m = fro()
    bou = fmod("boussinesq")
    nat = fmod("natural_conv")
    N = 16 if quick else 24
    t_end = 0.4 if quick else 1.0
    presets = [1e4, 1e5] if quick else [1e4, 3e4, 1e5]
    runs = []
    for Ra in presets:
        out = bou.simulate_boussinesq(Ra=Ra, n_cells=N, t_end=t_end,
                                      collect_trace=True)
        traj = np.asarray(out["trajectory"], dtype=float)
        fr, _idx = _thin_traj(traj, 21 if quick else 41)
        emb = fr[:, ::2, ::2]
        dx = 1.0 / N
        uc = (np.asarray(out["u"], dtype=float)[:-1, :]
              + np.asarray(out["u"], dtype=float)[1:, :]) / 2.0
        vc = (np.asarray(out["v"], dtype=float)[:, :-1]
              + np.asarray(out["v"], dtype=float)[:, 1:]) / 2.0
        st = 2
        gy, gx = np.mgrid[st // 2:N:st, st // 2:N:st]
        gx, gy = gx.ravel(), gy.ravel()
        umax = max(float(np.abs(uc).max()), float(np.abs(vc).max()), 1e-12)
        glyphs = [[round(float(gx[k] + 0.5) * dx, 4), round(float(gy[k] + 0.5) * dx, 4),
                   round(float(uc[gy[k], gx[k]]) / umax, 3),
                   round(float(vc[gy[k], gx[k]]) / umax, 3)]
                  for k in range(gx.size)]
        nu_res = float(out["Nu"])
        nu_corr = float(nat.nusselt(jnp.asarray(Ra), 0.707))
        runs.append({
            "label": f"Ra = {Ra:.0e}",
            "frames": [[[round(float(v), 4) for v in row] for row in f] for f in emb],
            "glyphs": glyphs, "umax": round(umax, 4),
            "Nu": round(nu_res, 4), "Nu_corr": round(nu_corr, 3),
            "regime": nat.regime_of(float(Ra)),
            "valid": bool(Ra <= 3e5 and N >= 24),
            "n": int(emb.shape[1]),
        })
    return {"runs": runs, "N": N, "t_end": t_end, "Pr": 0.707,
            "n_snap": len(runs[0]["frames"])}


def run_cloak(*, quick: bool = False) -> dict:
    """Ring-cloak presets: T movies + kappa minimap + objectives."""
    import jax  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)
    m = fro()
    clk = fmod("cloak2d")
    N = 24 if quick else 32
    # t_end=8 ≈ 0.8 diffusion times (L²/π²bg): fields developed enough that
    # the objectives discriminate (at t≪1 the cold interior scores trivially).
    t_end = 8.0
    presets = [(0.01, 0.01, "bare plate"), (0.1, 1.0, "naive ring (frozen defaults)"),
               (1e-4, 1.0, "insulating-core cloak")] if not quick else [
               (0.01, 0.01, "bare plate"), (1e-4, 1.0, "insulating-core cloak")]
    runs, bare_final = [], None
    for inner, ring, label in presets:
        kmax = max(inner, ring, 0.01)
        dt = 0.9 * clk.max_timestep_cloak(1.0 / N, 1.0 / N, kmax)
        n_steps = int(round(t_end / dt))
        out = clk.simulate_cloak(kappa_inner=inner, kappa_ring=ring, nx=N,
                                 ny=N, dt=dt, n_steps=n_steps, collect_trace=True)
        met = clk.cloak_metrics(out["final"], out["kappa"], out["X"], out["Y"],
                                out["R"], out["dx"], out["dy"], 1.0, inner, ring)
        if bare_final is None:
            bare_final = np.asarray(out["final"], dtype=float)
        import jax.numpy as jnp  # noqa: PLC0415

        R = np.asarray(out["R"], dtype=float)
        ext = R >= 0.4
        ff_delta = float(((np.asarray(out["final"], dtype=float) - bare_final)[ext] ** 2).mean())
        traj = np.asarray(out["trajectory"], dtype=float)
        fr, _idx = _thin_traj(traj, 21 if quick else 41)
        emb = fr[:, ::2, ::2]
        kap = np.asarray(out["kappa"], dtype=float)[::2, ::2]
        runs.append({
            "label": f"{label} (in {inner:g}, ring {ring:g})",
            "frames": [[[round(float(v), 4) for v in row] for row in f] for f in emb],
            "kappa": [[round(float(v), 4) for v in row] for row in kap],
            "grad": round(float(met["interior_grad_norm"]), 5),
            "mismatch": round(float(met["far_field_mismatch"]), 6),
            "ff_delta": round(ff_delta, 6),
            "klo": round(float(kap.min()), 5), "khi": round(float(kap.max()), 4),
            "n": int(emb.shape[1]),
        })
    return {"runs": runs, "N": N, "r_in": 0.2, "r_out": 0.4,
            "n_snap": len(runs[0]["frames"]), "t_end": t_end}


def run_telegrapher(*, quick: bool = False) -> dict:
    """Thermal pulse at damping presets (finite-speed proof included)."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)
    m = fro()
    N = 32 if quick else 48
    grid = m["dh"].Grid2D.uniform(Lx=2.0, Ly=2.0, nx=N, ny=N)
    bc = m["dh"].BoundaryCondition2D(
        left={"kind": "dirichlet", "value": 0.0},
        right={"kind": "dirichlet", "value": 0.0},
        bottom={"kind": "dirichlet", "value": 0.0},
        top={"kind": "dirichlet", "value": 0.0})
    X, Y = np.asarray(grid.X.T, dtype=float), np.asarray(grid.Y.T, dtype=float)
    u0 = np.exp(-((X - 1.0) ** 2 + (Y - 1.0) ** 2) / 0.02)
    v0 = np.zeros_like(u0)
    r_grid = np.sqrt((X - 1.0) ** 2 + (Y - 1.0) ** 2)
    dx = 2.0 / N
    runs = []
    for tau, label in ([(0.5, "wave"), (0.05, "damped")] if quick
                       else [(0.5, "wave"), (0.05, "damped"), (0.005, "diffusive")]):
        eqn = m["dh"].TelegrapherEquation2D(grid=grid, bc=bc, alpha=1.0, tau=tau)
        dt = min(0.002, 0.9 * dx * dx / 4.0)
        traj = m["dh"].solve_telegrapher_2d(eqn, jnp.asarray(u0), jnp.asarray(v0),
                                            (0.0, 0.3), dt)
        fr, _idx = _thin_traj(np.asarray(traj, dtype=float), 21 if quick else 41)
        emb = fr[:, ::2, ::2]
        v = float(np.sqrt(1.0 / tau))
        peak = [round(float(f.max()), 5) for f in emb]
        mid = emb[len(emb) // 4]
        far = float(np.abs(mid[r_grid[::2, ::2] > 0.5]).max())
        runs.append({
            "label": f"τ = {tau:g} ({label}, v = {v:.2f})",
            "frames": [[[round(float(vv), 4) for vv in row] for row in f] for f in emb],
            "peak": peak, "v": round(v, 3), "far_mid": round(far, 8),
            "n": int(emb.shape[1]),
        })
    return {"runs": runs, "N": N, "kappa": 1.0, "n_snap": len(runs[0]["frames"])}


def run_te(*, quick: bool = False) -> dict:
    """Lumped pair curves (frozen) + resolved leg profiles (frozen te_1d)."""
    import jax  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)
    m = fro()
    fte = fmod("thermoelectric")
    te1d = fmod("te_1d")
    pairs = ["Bi2Te3", "PbTe", "SiGe"]
    currents = [round(0.5 * i, 2) for i in range(17)]
    curves = {}
    for name in pairs:
        qc, cop, oow = [], [], []
        for I in currents:
            r = fte.simulate_te(Tc_C=18.0, Th_C=35.0, current_A=I, pair=name)
            qc.append(round(float(r["Qc_W"]), 5))
            cop.append(round(float(r["COP"]), 5))
            oow.append(bool(float(r["out_of_window"]) > 0.5))
        p = fte.resolve_te_pair(name)
        curves[name] = {"I": currents, "Qc": qc, "COP": cop, "oow": oow,
                        "S_uV_K": p["S_V_K"] * 1e6, "R_mohm": p["R_ohm"] * 1e3,
                        "K_mW_K": p["K_W_K"] * 1e3, "Th_max_C": p["Th_max_C"],
                        "I_opt": round(float(fte.simulate_te(Tc_C=18.0, Th_C=35.0,
                                                             current_A=3.0, pair=name)["I_opt_A"]), 3)}
    profs = []
    for I in [1.0, 3.0, 5.0]:
        r = te1d.simulate_te_1d(current_A=I, n_cells=16, n_steps=3000)
        profs.append({"label": f"I = {I:g} A",
                      "T_C": [round(float(v) - 273.15, 3)
                              for v in np.asarray(r["T_final"], dtype=float)],
                      "COP": round(float(r["COP"]), 4),
                      "Qc": round(float(r["Qc_W"]), 5)})
    lump = fte.simulate_te(Tc_C=18.0, Th_C=35.0, current_A=3.0, pair="Bi2Te3")
    return {"curves": curves, "profiles": profs,
            "parity": {"leg_COP": profs[1]["COP"],
                       "lumped_COP": round(float(lump["COP"]), 4)},
            "Tc_C": 18.0, "Th_C": 35.0}


def run_absorption(*, quick: bool = False) -> dict:
    """Single-effect chiller curves over T_gen (live tree core)."""
    import numpy as np  # noqa: PLC0415

    import harness  # noqa: PLC0415

    pairs = ["LiBr-H2O", "NH3-H2O", "LiCl-H2O"]
    t_grid = [60.0 + 2.5 * i for i in range(15)]
    curves = {}
    for pair in pairs:
        env = harness.make("AbsorptionCycle-v0", material=pair, profile="human")
        cops, scps, dxs = [], [], []
        for tg in t_grid:
            r = env.evaluate({"t_gen_c": tg})
            cops.append(round(float(r["COP"]), 4))
            scps.append(round(float(r["SCP_W_kg"]), 2))
            dxs.append(round(float(r["delta_x"]), 5))
        from harness.physics import absorption as ab  # noqa: PLC0415

        p = ab.resolve_pair(pair)
        curves[pair] = {"t_gen": t_grid, "COP": cops, "SCP": scps, "dx": dxs,
                        "eps": p["eps"], "dx_max": p["dx_max"]}
    from harness.physics import absorption as ab  # noqa: PLC0415

    anchor = ab.simulate_absorption(t_evap_c=7.0, t_cond_c=30.0, t_gen_c=85.0,
                                    pair="LiBr-H2O")
    prof = harness.get_profile("human")
    return {"curves": curves,
            "anchor": {"COP": round(float(anchor["COP"]), 3),
                       "note": "V-ABS gate: LiBr-H2O at 7/30/85 °C ≈ 0.70 ±15%"},
            "t_evap_c": prof.t_evap_c, "t_cond_c": prof.t_cond_c,
            "cycle_time_s": prof.cycle_time_s}


def run_coldspot(*, quick: bool = False) -> dict:
    """Hot-walled plate with/without a localized sink (tracked avenue)."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)
    m = fro()
    from diffheat.mesh.boundary import apply_boundary_conditions_2d  # noqa: PLC0415
    from diffheat.operators import laplacian_2d  # noqa: PLC0415

    N = 24 if quick else 40
    grid = m["dh"].Grid2D.uniform(Lx=1.0, Ly=1.0, nx=N, ny=N)
    bc = m["dh"].BoundaryCondition2D(
        left={"kind": "dirichlet", "value": 1.0},
        right={"kind": "dirichlet", "value": 1.0},
        bottom={"kind": "dirichlet", "value": 1.0},
        top={"kind": "dirichlet", "value": 1.0})
    X, Y = np.asarray(grid.X.T, dtype=float), np.asarray(grid.Y.T, dtype=float)
    r2 = (X - 0.5) ** 2 + (Y - 0.5) ** 2
    mask = r2 < 0.15 ** 2
    alpha = 0.01

    def run(sink_amp: float):
        def rhs(T, _grid, t, _p):
            L_T, b = apply_boundary_conditions_2d(
                lambda x: laplacian_2d(x, grid), grid, bc, T)
            src = -sink_amp * jnp.exp(-r2 / (2 * 0.08 ** 2)) if sink_amp else 0.0
            return alpha * (L_T + b) + src

        traj = m["dhs"].solve_2d(rhs, jnp.full((N, N), 1.0), grid,
                                 (0.0, 3.0), 0.002)
        fr, _idx = _thin_traj(np.asarray(traj, dtype=float), 21 if quick else 41)
        return fr[:, ::2, ::2]

    base, ctrl = run(0.0), run(2.0)
    tmean = lambda f: round(float(f[-1][mask[::2, ::2]].mean()), 4)
    return {"n": int(base.shape[1]),
            "runs": [
                {"label": "hot walls, no control",
                 "frames": [[[round(float(v), 4) for v in row] for row in f] for f in base],
                 "target_mean": tmean(base)},
                {"label": "hot walls + localized sink (hand-picked; optimizer deferred)",
                 "frames": [[[round(float(v), 4) for v in row] for row in f] for f in ctrl],
                 "target_mean": tmean(ctrl)}],
            "n_snap": len(base), "target_r": 0.15,
            "note": "Future avenue §10.2: control is hand-picked, not optimized."}


def run_junction(*, quick: bool = False) -> dict:
    """Two-material rod vs uniform rod steady drop (tracked avenue)."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)
    m = fro()
    N = 32 if quick else 64
    L = 1.0
    grid = m["dh"].Grid1D.uniform(L, N)
    from diffheat.mesh.boundary import BoundaryCondition  # noqa: PLC0415

    bc = BoundaryCondition(kind="dirichlet", value=jnp.array([1.0, 0.0]))
    xc = (np.arange(N, dtype=float) + 0.5) / N
    outs = {}
    for name, a in (("uniform", np.full(N, 0.01)),
                    ("junction (0.01 | 0.001)", np.where(xc < 0.5, 0.01, 0.001))):
        eqn = m["dhp"].HeatEquation1D(grid=grid, bc=bc, alpha=jnp.asarray(a))
        dt = 0.9 * float(jnp.min(grid.dx)) ** 2 / (2 * float(a.max()))
        traj = m["dhs"].solve_heat_1d(eqn, jnp.zeros(N), (0.0, 30.0), dt)
        fr, idx = _thin_traj(np.asarray(traj, dtype=float), 21 if quick else 41)
        fin = fr[-1]
        j = slice(N // 2 - 2, N // 2 + 2)
        outs[name] = {"frames": [[round(float(v), 4) for v in row] for row in fr],
                      "drop": round(float(fin[j.start] - fin[j.stop - 1]), 4),
                      "dt": dt}
    return {"xs": xc.tolist(), "runs": outs,
            "t": [round(float(i) * outs["uniform"]["dt"], 3) for i in idx],
            "note": "Future avenue §10.3: steady ΔT proxy across the junction."}


def run_metamaterial(*, quick: bool = False) -> dict:
    """Grid-search pass over ring parameters through the frozen solver."""
    import itertools  # noqa: PLC0415

    import jax  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)
    m = fro()
    clk = fmod("cloak2d")
    N = 20
    t_end = 4.0 if quick else 8.0
    dt = 0.9 * clk.max_timestep_cloak(1.0 / N, 1.0 / N, 5.0)
    n_steps = int(round(t_end / dt))
    table = []
    for inner, ring in itertools.product([1e-4, 0.01, 0.1], [0.05, 0.5, 1.0, 5.0]):
        out = clk.simulate_cloak(kappa_inner=inner, kappa_ring=ring, nx=N,
                                 ny=N, dt=dt, n_steps=n_steps)
        met = clk.cloak_metrics(out["final"], out["kappa"], out["X"], out["Y"],
                                out["R"], out["dx"], out["dy"], 1.0, inner, ring)
        obj = float(met["interior_grad_norm"]) + 0.1 * float(met["far_field_mismatch"])
        table.append({"inner": inner, "ring": ring, "obj": round(obj, 5),
                      "grad": round(float(met["interior_grad_norm"]), 5),
                      "mismatch": round(float(met["far_field_mismatch"]), 6)})
    table.sort(key=lambda r: r["obj"])
    best = table[0]
    out = clk.simulate_cloak(kappa_inner=best["inner"], kappa_ring=best["ring"],
                             nx=N, ny=N, dt=dt, n_steps=n_steps, collect_trace=True)
    traj = np.asarray(out["trajectory"], dtype=float)
    fr, _idx = _thin_traj(traj, 21 if quick else 41)
    emb = fr[:, ::2, ::2] if N > 20 else fr
    kap = np.asarray(out["kappa"], dtype=float)
    kap = kap[::2, ::2] if N > 20 else kap
    return {"table": table, "best": best, "n": int(emb.shape[1]),
            "frames": [[[round(float(v), 4) for v in row] for row in f] for f in emb],
            "kappa": [[round(float(v), 4) for v in row] for row in kap],
            "n_snap": len(emb),
            "note": "Future avenue §10.7: grid search through the frozen rollout; gradient descent deferred."}


__all__ = ["CACHE", "FROZEN_SHA", "FROZEN_SHORT", "PDE_PAGES", "frozen_pkg", "fro",
           "run_absorption", "run_boussinesq", "run_cloak", "run_coldspot",
           "run_disc", "run_forced", "run_heat1d", "run_heat2d", "run_heat3d",
           "run_junction", "run_metamaterial", "run_te", "run_telegrapher",
           "write_pde_pages", "fmod", "chrome", "runs_page", "page_script"]


# --------------------------------------------------------------------------
# Shared SVG/JS builders (standalone pages — no cross-file script deps).
# --------------------------------------------------------------------------

def chrome(title: str, body: str, script: str) -> str:
    from .viz_design import CSS_ANIM  # noqa: PLC0415 (lazy, light module)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
body {{ background:{_BG}; color:{_FG}; font-family:system-ui,sans-serif; max-width:760px; margin:0 auto; padding:16px; }}
h1 {{ font-size:22px; }} h2 {{ font-size:16px; margin-top:28px; }}
{CSS_ANIM}
.presets button {{ background:{_PNL}; color:{_FG}; border:1px solid {_SPN}; border-radius:6px; padding:4px 10px; margin:2px; cursor:pointer; font-size:12px; }}
.controls {{ display:flex; gap:10px; align-items:center; margin:10px 0; flex-wrap:wrap; font-size:13px; }}
select, input[type=range] {{ accent-color:{_AMB}; }} select {{ background:{_PNL}; color:{_FG}; border:1px solid {_SPN}; border-radius:6px; padding:4px; }}
button {{ background:{_PNL}; color:{_FG}; border:1px solid {_SPN}; border-radius:6px; padding:4px 10px; cursor:pointer; }}
.meters span {{ display:inline-block; background:{_PNL}; border:1px solid {_SPN}; border-radius:6px; padding:3px 9px; margin:2px; font-size:12px; }}
table {{ border-collapse:collapse; width:100%; font-size:12px; }} td, th {{ border:1px solid {_SPN}; padding:4px 6px; text-align:left; }}
th {{ color:{_AX}; }} .dim {{ color:{_AX}; font-size:12px; }} a {{ color:{_CYN}; }}
.math {{ background:{_PNL}; border:1px solid {_SPN}; border-radius:8px; padding:8px 12px; font-size:13px; }}
</style>
</head>
<body>
{body}
<script>
{JS_CMAP}
{JS_FIELD}
{script}
</script>
</body>
</html>"""


def _svg_open(w: int, h: int) -> str:
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" '
            f'style="background:{_BG};border:1px solid {_SPN};border-radius:8px">')


def field_grid_svg(p: str, nx: int, ny: int, w: int = 720, h: int = 340,
                   title: str = "", ox: int = 56, oy: int = 14,
                   overlay_gid: str | None = None) -> str:
    """Rect grid skeleton, ids ``{p}-{i}-{j}`` (i = row/y, j = col/x)."""
    pw, ph = w - ox - 16, h - oy - 52
    cw, ch = pw / nx, ph / ny
    s = [_svg_open(w, h)]
    if title:
        s.append(f'<text x="16" y="24" fill="{_FG}" font-size="13" '
                 f'font-weight="bold">{title}</text>')
    for i in range(ny):
        for j in range(nx):
            s.append(f'<rect id="{p}-{i}-{j}" x="{ox + j * cw:.1f}" y="{oy + i * ch:.1f}" '
                     f'width="{cw + 0.4:.1f}" height="{ch + 0.4:.1f}" fill="#141428"/>')
    if overlay_gid:
        s.append(f'<g id="{overlay_gid}"></g>')
    s.append(f'<text x="{ox}" y="{h - 8}" fill="{_AX}" font-size="10">{nx}×{ny} cells</text>')
    return "\n".join(s) + "\n</svg>"


def glyph_js() -> str:
    return """
function rebuildGlyphs(gid, glyphs, ox, oy, pw, ph, scale) {
  const g = document.getElementById(gid);
  let s = "";
  glyphs.forEach(([x, y, u, v]) => {
    const x0 = ox + pw * x, y0 = oy + ph * y;
    const x1 = x0 + scale * u, y1 = y0 + scale * v;
    s += '<line x1="' + x0.toFixed(1) + '" y1="' + y0.toFixed(1) + '" x2="' + x1.toFixed(1) +
         '" y2="' + y1.toFixed(1) + '" stroke="#e2e8f0" stroke-width="1.4"/>';
  });
  g.innerHTML = s;
}
"""


def runs_page(*, title: str, intro: str, math: str, runs: list[dict],
              prov: str, extra: str = "",
              glyph_box: dict | None = None, minimap: dict | None = None,
              closing: str = "Presets are separate frozen-solver runs, never interpolations.") -> str:
    """Generic multi-preset field page body (script assembled by caller)."""
    btns = "\n".join(
        f'<button id="run-{k}" style="border-color:{_AMB if k == 0 else _SPN}">{r["label"]}</button>'
        for k, r in enumerate(runs))
    meters = "\n".join(
        f'<div class="meters" id="meters-{k}" style="display:{"block" if k == 0 else "none"}">' +
        "".join(f"<span>{kk}: <b>{vv}</b></span>" for kk, vv in r["meters"].items()) + "</div>"
        for k, r in enumerate(runs))
    grids = runs[0]["grids"]
    grids_html = ""
    for gi, g in enumerate(grids):
        grids_html += field_grid_svg(
            g["p"], g["nx"], g["ny"], title=g.get("title", ""),
            overlay_gid=glyph_box["gid"] if (glyph_box and gi == 0) else None) + "\n"
    mini_html = ""
    if minimap:
        m = minimap
        mini_html = field_grid_svg(m["p"], m["nx"], m["ny"], h=180, title=m["title"])
    n = len(runs[0]["grids"][0]["frames"])
    return f"""
<p class="dim"><a href="index.html">← all applications</a> · {prov}</p>
<h1>{title}</h1>
<p class="dim">{intro}</p>
<div class="math">{math}</div>
<div class="presets">{"".join(btns)}</div>
{meters}
<div class="controls"><button id="pp-play">▶ play</button>
<input id="pp-slider" type="range" min="0" max="{n - 1}" value="0" step="1">
<span class="dim" id="pp-t"></span></div>
{grids_html}
{mini_html}
{extra}
<p class="dim">{closing}</p>"""


def page_script(runs: list[dict], t_vals: list,
                glyph_box: dict | None = None, minimap: dict | None = None) -> str:
    """Generic preset/scrub/play wiring over ``RUNS`` (times embedded)."""
    glyph = ""
    if glyph_box:
        g = glyph_box
        glyph = (glyph_js() + f"\nconst GLYPH = {json.dumps(g)};\n"
                 "function drawGlyphs(){ const r = RUNS[cur]; if (r.glyphs) "
                 f"rebuildGlyphs(GLYPH.gid, r.glyphs, {g['ox']}, {g['oy']}, {g['pw']}, {g['ph']}, {g['scale']}); }}\n")
    mini = ""
    if minimap:
        m = minimap
        mini = (f"\nfunction drawMinimap(){{ const r = RUNS[cur]; "
                f"paintGrid('{m['p']}', r.kfield, r.klo, r.khi); }}\n")
    return f"""const RUNS = {json.dumps(runs)};
const TIMES = {json.dumps(t_vals)};
let cur = 0;
function paintAll(f) {{
  const r = RUNS[cur];
  r.grids.forEach((g) => paintGrid(g.p, g.frames[f], r.lo, r.hi));
  document.getElementById("pp-t").textContent = "t = " + TIMES[f];
}}
function selectRun(k) {{
  cur = k;
  RUNS.forEach((_, q) => {{
    document.getElementById("run-" + q).style.borderColor = q === k ? "#f59e0b" : "#1e222a";
    document.getElementById("meters-" + q).style.display = q === k ? "block" : "none";
  }});
  {("drawGlyphs();" if glyph_box else "")}
  {("drawMinimap();" if minimap else "")}
  document.getElementById("pp-slider").value = 0;
  paintAll(0);
}}
function initRuns() {{
  RUNS.forEach((_, k) => document.getElementById("run-" + k).addEventListener("click", () => selectRun(k)));
  wireScrub("pp-slider", "pp-play", RUNS[0].grids[0].frames.length, paintAll);
  selectRun(0);
}}
document.addEventListener("DOMContentLoaded", initRuns);
{glyph}{mini}"""


def prov_str(note: str) -> str:
    return (f"frozen {FROZEN_SHORT} ({FROZEN_SHA[:12]}…), generated "
            f"{datetime.date.today().isoformat()} · {note}")


def _run_dict(label: str, grids: list[dict], lo: float, hi: float,
              meters: dict, **kw) -> dict:
    d = {"label": label, "grids": grids, "lo": lo, "hi": hi, "meters": meters}
    d.update(kw)
    return d


def _grid(p: str, frames, nx: int, ny: int, title: str = "") -> dict:
    return {"p": p, "nx": nx, "ny": ny, "frames": frames, "title": title}


def page_heat1d(d: dict) -> str:
    n = len(d["frames"])
    steady = [d["steady"]] * n
    runs = [_run_dict(
        f"slab L={d['L']:g}, α={d['alpha']:g}, N={d['N']}",
        [_grid("h1", d["frames"], d["N"], 1, "T(x,t) — Dirichlet ends 1 / 0"),
         _grid("h1s", steady, d["N"], 1, "steady state 1−x (analytic line)")],
        0.0, 1.0,
        {"λ₁ numeric": f"{d['lambda1']:.4f}", "λ₁ exact π²": f"{d['lambda1_exact']:.4f}",
         "rel. err": f"{abs(d['lambda1'] - d['lambda1_exact']) / d['lambda1_exact']:.2e}",
         "dt": f"{d['dt']:.2e}", "t_end": f"{d['t_end']:g}"})]
    body = runs_page(title="Heat 1D — transient slab", intro="",
                     math="∂T/∂t = α ∂²T/∂x², T(0)=1, T(1)=0. The numeric replay runs against the analytic steady line; the decay rate is checked against the first Laplacian eigenvalue.",
                     runs=runs, prov=prov_str(f"diffheat solve_heat_1d, N={d['N']}"))
    return chrome("Heat 1D — transient slab (frozen diffheat)", body,
                  page_script(runs, d["t"]))


def page_heat2d(d: dict) -> str:
    runs = [_run_dict(
        f"plate 1×1, α={d['alpha']:g}, solved {d['solved_n']}²",
        [_grid("h2", d["frames"], d["nx"], d["ny"], "T(x,y,t) — hot left, cold right, insulated top/bottom")],
        0.0, 100.0,
        {"final mean": f"{d['final_mean']:.1f} °C", "steady mean": f"{d['steady_mean']:.1f} °C",
         "α": f"{d['alpha']:g}"})]
    body = runs_page(title="Heat 2D — hot/cold plate", intro="",
                     math="∂T/∂t = α∇²T. Square plate from 0 °C with left edge 100 °C, right edge 0 °C: the field approaches the linear steady gradient (mean 50 °C).",
                     runs=runs, prov=prov_str(f"diffheat solve_heat_2d, {d['solved_n']}², embedded {d['nx']}²"))
    return chrome("Heat 2D — plate (frozen diffheat)", body, page_script(runs, d["t"]))


def page_heat3d(d: dict) -> str:
    n = d["N"]
    runs = [_run_dict(
        f"cube 1³, N={n}",
        [_grid("sxy", d["slices"]["xy"], n, n, "mid z-plane"),
         _grid("sxz", d["slices"]["xz"], n, n, "mid y-plane"),
         _grid("syz", d["slices"]["yz"], n, n, "mid x-plane")],
        0.0, 100.0, {"N": f"{n}³", "BCs": "hot/cold x-faces, insulated rest"})]
    body = runs_page(title="Heat 3D — cube slices", intro="",
                     math="∂T/∂t = α∇²T in the cube; three orthogonal mid-planes animate together.",
                     runs=runs, prov=prov_str(f"diffheat solve_heat_3d, {n}³"))
    return chrome("Heat 3D — cube (frozen diffheat)", body, page_script(runs, d["t"]))
    """Annular-sector skeleton, ids ``dc-{r}-{s}``."""
    import math  # noqa: PLC0415

    cx, cy, Rpx = 360, 175, 150
    s = [_svg_open(720, 340)]
    s.append(f'<text x="16" y="24" fill="{_FG}" font-size="13" font-weight="bold">'
             f'T(r,θ,t) — exact Bessel series</text>')
    for r in range(nr):
        r0, r1 = Rpx * r / nr, Rpx * (r + 1) / nr
        for sec in range(nsec):
            a0 = 2 * math.pi * sec / nsec
            a1 = 2 * math.pi * (sec + 1) / nsec
            x0, y0 = cx + r0 * math.cos(a0), cy + r0 * math.sin(a0)
            x1, y1 = cx + r1 * math.cos(a0), cy + r1 * math.sin(a0)
            x2, y2 = cx + r1 * math.cos(a1), cy + r1 * math.sin(a1)
            x3, y3 = cx + r0 * math.cos(a1), cy + r0 * math.sin(a1)
            s.append(f'<path id="dc-{r}-{sec}" d="M{x0:.1f},{y0:.1f} L{x1:.1f},{y1:.1f} '
                     f'A{r1:.1f},{r1:.1f} 0 0 1 {x2:.1f},{y2:.1f} L{x3:.1f},{y3:.1f} '
                     f'A{r0:.1f},{r0:.1f} 0 0 0 {x0:.1f},{y0:.1f} Z" fill="#141428"/>')
    return "\n".join(s) + "\n</svg>"


def disc_svg(nr: int, nsec: int) -> str:
    """Annular-sector skeleton, ids ``dc-{r}-{s}``."""
    import math  # noqa: PLC0415

    cx, cy, Rpx = 360, 175, 150
    s = [_svg_open(720, 340)]
    s.append(f'<text x="16" y="24" fill="{_FG}" font-size="13" font-weight="bold">'
             f'T(r,θ,t) — exact Bessel series</text>')
    for r in range(nr):
        r0, r1 = Rpx * r / nr, Rpx * (r + 1) / nr
        for sec in range(nsec):
            a0 = 2 * math.pi * sec / nsec
            a1 = 2 * math.pi * (sec + 1) / nsec
            x0, y0 = cx + r0 * math.cos(a0), cy + r0 * math.sin(a0)
            x1, y1 = cx + r1 * math.cos(a0), cy + r1 * math.sin(a0)
            x2, y2 = cx + r1 * math.cos(a1), cy + r1 * math.sin(a1)
            x3, y3 = cx + r0 * math.cos(a1), cy + r0 * math.sin(a1)
            s.append(f'<path id="dc-{r}-{sec}" d="M{x0:.1f},{y0:.1f} L{x1:.1f},{y1:.1f} '
                     f'A{r1:.1f},{r1:.1f} 0 0 1 {x2:.1f},{y2:.1f} L{x3:.1f},{y3:.1f} '
                     f'A{r0:.1f},{r0:.1f} 0 0 0 {x0:.1f},{y0:.1f} Z" fill="#141428"/>')
    return "\n".join(s) + "\n</svg>"


def page_disc(d: dict) -> str:
    body = f"""
<p class="dim"><a href="index.html">← all applications</a> · {prov_str("diffheat analytic Bessel series (exact, not a discretization)")}</p>
<h1>Heat on a disc — analytic</h1>
<p class="dim">u_t = ∇²u on R=1 with u=0 on the rim, from an off-center bump. The Bessel eigenfunction series <b>is</b> the solution — no grid, no CFL, no numerical diffusion. (Curved boundaries need no unstructured mesh here: the polar basis fits the domain.)</p>
<div class="math">u(r,θ,t) = Σ<sub>m,n</sub> v<sub>m,n</sub>(r,θ) · c<sub>m,n</sub> · exp(−λ<sub>m,n</sub> t)</div>
<div class="meters"><span>peak now: <b id="dc-peak">–</b></span></div>
<div class="controls"><button id="pp-play">▶ play</button>
<input id="pp-slider" type="range" min="0" max="{len(d['frames']) - 1}" value="0" step="1">
<span class="dim" id="pp-t"></span></div>
{disc_svg(d["nr"], d["nsec"])}"""
    script = f"""const FR = {json.dumps(d["frames"])};
const TS = {json.dumps(d["t"])};
const PEAK = {json.dumps(d["peak"])};
const NR = {d["nr"]}, NS = {d["nsec"]};
function paintDisc(f) {{
  const F = FR[f], lo = 0, hi = Math.max(...F.flat(), 1e-9);
  for (let r = 0; r < NR; r++) for (let s = 0; s < NS; s++)
    document.getElementById("dc-" + r + "-" + s).setAttribute("fill", inferno(F[r][s] / hi));
  document.getElementById("pp-t").textContent = "t = " + TS[f];
  document.getElementById("dc-peak").textContent = PEAK[f].toFixed(4);
}}
function initDisc() {{
  wireScrub("pp-slider", "pp-play", FR.length, paintDisc);
  paintDisc(0);
}}
document.addEventListener("DOMContentLoaded", initDisc);"""
    return chrome("Heat on a disc — analytic (frozen diffheat)", body, script)


    return chrome("Forced convection — channel (frozen diffheat)", body,
                  page_script(runs, d["t"],
                              glyph_box={"gid": "fc-g", "ox": 56, "oy": 14, "pw": 648, "ph": 274, "scale": 22}))


def page_forced(d: dict) -> str:
    nx, ny = d["nx"], d["ny"]
    glyphs = [[round(j / (nx - 1), 4), round(i / (ny - 1), 4), 1.0, 0.0]
              for i in range(1, ny, 2) for j in range(2, nx, 4)]
    runs = [_run_dict(
        f"channel {d['Lx']:g}×{d['Ly']:g}, U={d['U']:g} (Pe={d['Pe']:.0f})",
        [_grid("fc", d["frames"], nx, ny, "T(x,y,t) — cold inlet sweeps the heated chip")],
        0.0, d["tmax"],
        {"Pe = UL/α": f"{d['Pe']:.0f} (advection-dominated)" if d["Pe"] > 10 else f"{d['Pe']:.0f}",
         "chip peak": f"{d['tmax']:.1f} °C", "U": f"{d['U']:g} m/s"},
        glyphs=glyphs)]
    body = runs_page(title="Forced convection — channel + chip", intro="",
                     math="∂T/∂t + u·∇T = α∇²T with prescribed uniform flow (one-way coupling). Cold inlet, insulated walls, Gaussian chip source at "
                     f"({d['chip_x0']:g}, {d['chip_y0']:g}). Glyphs show the prescribed velocity.",
                     runs=runs, prov=prov_str(f"frozen forced-conv rollout, {nx * 2}×{ny * 2} solved"),
                     glyph_box={"gid": "fc-g", "ox": 56, "oy": 14, "pw": 648, "ph": 274, "scale": 22})
    return chrome("Forced convection — channel (frozen diffheat)", body,
                  page_script(runs, d["t"],
                              glyph_box={"gid": "fc-g", "ox": 56, "oy": 14, "pw": 648, "ph": 274, "scale": 22}))


def page_boussinesq(d: dict) -> str:
    runs = []
    for r in d["runs"]:
        runs.append(_run_dict(
            r["label"], [_grid("bq", r["frames"], r["n"], r["n"], "T(x,y,t) — hot bottom, cold top")],
            0.0, 1.0,
            {"Nu resolved": f"{r['Nu']:.3f}", "Nu correlation": f"{r['Nu_corr']:.3f}",
             "regime": r["regime"], "validated envelope": "yes" if r["valid"] else "NO — extrapolated",
             "u_max": f"{r['umax']:.3f}"},
            glyphs=r["glyphs"]))
    body = runs_page(title="Boussinesq cavity — convection onset", intro="",
                     math="Nondimensional Boussinesq set (MAC grid, artificial compressibility, explicit): "
                     "u_t+(u·∇)u = −∇p+Pr∇²u; v_t+(u·∇)v = −∇p+Pr∇²v+Ra·Pr·T; T_t+(u·∇)T = ∇²T. "
                     "Glyphs are the solved velocity at end of run. The correlation meter is the frozen Globe–Dropkin Nu(Ra) for comparison.",
                     runs=runs, prov=prov_str(f"frozen boussinesq rollout, {d['N']}², t_end={d['t_end']:g} diffusion units"),
                     glyph_box={"gid": "bq-g", "ox": 56, "oy": 14, "pw": 648, "ph": 274, "scale": 26})
    t_vals = [round(i * d["t_end"] / (len(runs[0]["grids"][0]["frames"]) - 1), 4)
              for i in range(len(runs[0]["grids"][0]["frames"]))]
    return chrome("Boussinesq cavity (frozen diffheat)", body,
                  page_script(runs, t_vals,
                              glyph_box={"gid": "bq-g", "ox": 56, "oy": 14, "pw": 648, "ph": 274, "scale": 26}))


def page_cloak(d: dict) -> str:
    runs = []
    for r in d["runs"]:
        runs.append(_run_dict(
            r["label"], [_grid("ck", r["frames"], r["n"], r["n"], "T(x,y,t) — hot left, cold right")],
            0.0, 1.0,
            {"interior |∇T|": f"{r['grad']:.4f} (want ~0)",
             "far-field mismatch": f"{r['mismatch']:.2e} (want ~0)",
             "far-field Δ vs bare": f"{r['ff_delta']:.2e} (like-for-like)"},
            kfield=r["kappa"], klo=r["klo"], khi=r["khi"]))
    body = runs_page(title="Thermal cloak — ring design", intro="",
                     math="∂T/∂t = κ(x,y)·∇²T with a ring diffusivity field (inner / ring / background). "
                     "Objectives: shield the interior (r &lt; 0.2) while the exterior matches the linear gradient. "
                     "Lower map shows the κ field. Note the documented gap: field-multiply form, not full ∇·(κ∇T).",
                     runs=runs, prov=prov_str("frozen cloak2d rollout"),
                     minimap={"p": "ckk", "nx": len(runs[0]["kfield"]),
                              "ny": len(runs[0]["kfield"]), "title": "κ(x,y) diffusivity field"})
    t_vals = list(range(len(runs[0]["grids"][0]["frames"])))
    return chrome("Thermal cloak (frozen diffheat)", body,
                  page_script(runs, t_vals, minimap={"p": "ckk"}))


def page_telegrapher(d: dict) -> str:
    runs = []
    for r in d["runs"]:
        runs.append(_run_dict(
            r["label"], [_grid("tg", r["frames"], r["n"], r["n"], "u(x,y,t) — Gaussian pulse, absorbing rim")],
            0.0, 1.0,
            {"wave speed v": f"{r['v']:.2f}",
             "far field @t/4": f"{r['far_mid']:.1e} (≈0 proves finite speed)"}))
    body = runs_page(title="Telegrapher — thermal pulse", intro="",
                     math="τ·u_tt + u_t = κ∇²u. Large τ behaves as a wave (finite speed v=√(κ/τ): the far field stays exactly zero); "
                     "small τ recovers Fourier diffusion. The far-field meter checks max |u| beyond r=0.5 early in the run.",
                     runs=runs, prov=prov_str("diffheat telegrapher solver"))
    t_vals = list(range(len(runs[0]["grids"][0]["frames"])))
    return chrome("Telegrapher pulse (frozen diffheat)", body, page_script(runs, t_vals))


def page_coldspot(d: dict) -> str:
    runs = []
    for r in d["runs"]:
        n = r["frames"][0].__len__()
        runs.append(_run_dict(
            r["label"], [_grid("cs", r["frames"], d["n"], d["n"], "T(x,y,t) — all walls hot (1.0)")],
            0.0, 1.0, {"target-disc mean": f"{r['target_mean']:.3f}"}))
    body = runs_page(title="Hot-driven cold spot — boundary control", intro="",
                     math="Tracked future avenue (§10.2): all walls held hot. A localized Gaussian sink carves a sub-ambient spot in the target disc "
                     f"(r={d['target_r']:g}). Baseline vs hand-picked control shown side by side as presets — the optimizer is deferred.",
                     runs=runs, prov=prov_str("diffheat heat-2d with source term"),
                     closing="Control is hand-picked, not optimized — first simulation of the tracked avenue.")
    t_vals = list(range(len(runs[0]["grids"][0]["frames"])))
    return chrome("Cold spot control (frozen diffheat)", body, page_script(runs, t_vals))


def page_junction(d: dict) -> str:
    runs = [_run_dict(
        "rod L=1, ends 1/0",
        [_grid("ju", d["runs"]["uniform"]["frames"], len(d["xs"]), 1, "uniform α — linear steady state"),
         _grid("jj", d["runs"]["junction (0.01 | 0.001)"]["frames"], len(d["xs"]), 1,
               "junction α 0.01|0.001 — kinked steady state")],
        0.0, 1.0,
        {"ΔT across junction": f"{d['runs']['junction (0.01 | 0.001)']['drop']:.3f}",
         "ΔT uniform (same Δx)": f"{d['runs']['uniform']['drop']:.3f}"})]
    body = runs_page(title="Heat-to-work proxy — material junction", intro="",
                     math="Tracked future avenue (§10.3): steady temperature drop across a conductivity jump vs the uniform rod — "
                     "the proxy for harvestable ΔT at a junction.",
                     runs=runs, prov=prov_str("diffheat heat-1d with field α"))
    return chrome("Junction proxy (frozen diffheat)", body, page_script(runs, d["t"]))


def page_metamaterial(d: dict) -> str:
    rows = "\n".join(
        f"<tr{' style=\"background:#223042\"' if r == d['best'] else ''}><td>{r['inner']:g}</td><td>{r['ring']:g}</td>"
        f"<td>{r['obj']:.4f}</td><td>{r['grad']:.4f}</td><td>{r['mismatch']:.2e}</td></tr>"
        for r in d["table"])
    table = (f'<table><tr><th>κ inner</th><th>κ ring</th><th>objective</th><th>interior grad</th><th>mismatch</th></tr>'
             f'{rows}</table>')
    runs = [_run_dict(
        f"best: inner {d['best']['inner']:g}, ring {d['best']['ring']:g}",
        [_grid("mm", d["frames"], d["n"], d["n"], "T(x,y,t) — winning ring")],
        0.0, 1.0,
        {"objective": f"{d['best']['obj']:.4f}", "interior grad": f"{d['best']['grad']:.4f}",
         "mismatch": f"{d['best']['mismatch']:.2e}"},
        kfield=d["kappa"], klo=min(min(r) for r in d["kappa"]),
        khi=max(max(r) for r in d["kappa"]))]
    body = runs_page(title="Metamaterial — ring inverse design", intro="",
                     math="Tracked future avenue (§10.7): first grid-search pass over (κ inner, κ ring) through the frozen cloak rollout, "
                     "objective = interior |∇T| + 0.1·far-field mismatch. Gradient descent deferred.",
                     runs=runs, prov=prov_str("12 frozen cloak rollouts + winning replay"),
                     minimap={"p": "mmk", "nx": len(d["kappa"]), "ny": len(d["kappa"]),
                              "title": "winning κ(x,y)"},
                     extra=f"<h2>Search table</h2>{table}",
                     closing="Grid search, not gradients — the extension head is a differentiable descent.")
    t_vals = list(range(len(runs[0]["grids"][0]["frames"])))
    return chrome("Metamaterial ring search (frozen diffheat)", body,
                  page_script(runs, t_vals, minimap={"p": "mmk"}))


def curve_chart_svg(cid: str, series: list[dict], xlab: str, ylab: str,
                    w: int = 720, h: int = 300) -> str:
    """Static multi-polyline chart; JS moves crosshair group ``{cid}-x``."""
    s = [_svg_open(w, h)]
    s.append(f'<text x="16" y="24" fill="{_FG}" font-size="13" font-weight="bold">{xlab} → {ylab}</text>')
    s.append(f'<g id="{cid}-lines"></g><g id="{cid}-x"></g>')
    s.append(f'<text x="16" y="{h - 10}" fill="{_AX}" font-size="10">{xlab}</text>')
    s.append('</svg>')
    lines = []
    for se in series:
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in se["px"])
        dash = f' stroke-dasharray="{se["dash"]}"' if se.get("dash") else ""
        lines.append(f'<path d="M{pts.replace(" ", " L")}" fill="none" stroke="{se["color"]}" '
                     f'stroke-width="2"{dash}/>')
    out = "\n".join(s)
    out = out.replace(f'<g id="{cid}-lines"></g>', f'<g id="{cid}-lines">{"".join(lines)}</g>')
    return out


def _linmap(v: float, d0: float, d1: float, p0: float, p1: float) -> float:
    return p0 + (p1 - p0) * (v - d0) / max(d1 - d0, 1e-12)


def page_te(d: dict) -> str:
    x0, x1, y0, y1 = 56, 704, 40, 214
    I, Qc = d["curves"]["Bi2Te3"]["I"], None
    cols = {"Bi2Te3": _AMB, "PbTe": _CYN, "SiGe": _GRN}
    series = []
    for name, c in d["curves"].items():
        px = [(_linmap(i, 0, 8, x0, x1), _linmap(q, 0, max(max(cc["Qc"]) for cc in d["curves"].values()) * 1.05, y1, y0))
              for i, q in zip(c["I"], c["Qc"])]
        series.append({"px": px, "color": cols[name], "name": name})
    chart = curve_chart_svg("te", series, "current I [A]", "cooling Qc [W]")
    pairs_rows = "\n".join(
        f"<tr><td>{n}</td><td>{c['S_uV_K']:.0f}</td><td>{c['R_mohm']:.0f}</td>"
        f"<td>{c['K_mW_K']:.1f}</td><td>{c['Th_max_C']:.0f}</td><td>{c['I_opt']:.2f}</td></tr>"
        for n, c in d["curves"].items())
    prof = d["profiles"][1]
    leg = "".join(
        f'<rect id="te-leg-{k}" x="{56 + k * (648 / 16):.1f}" y="60" width="{648 / 16 - 1:.1f}" '
        f'height="60" fill="#141428"/>' for k in range(16))
    body = f"""
<p class="dim"><a href="index.html">← all applications</a> · {prov_str("frozen lumped pair model + frozen te_1d leg")}</p>
<h1>Thermoelectric — pair + leg</h1>
<p class="dim">Qc = S·Tc·I − ½I²R − K·ΔT per pair at Tc={d['Tc_C']:.0f} °C / Th={d['Th_C']:.0f} °C. Move the current slider: crosshair + readouts recompute live (closed forms). The leg strip is the resolved 1 mm segmented-leg steady profile at 3 A.</p>
<div class="math">Qc = S·Tc·I − ½·I²·R − K·ΔT · W<sub>in</sub> = S·I·ΔT + I²·R · COP = Qc/W<sub>in</sub> (0 when Qc ≤ 0)</div>
<div class="controls"><label>pair <select id="te-pair"><option>Bi2Te3</option><option>PbTe</option><option>SiGe</option></select></label>
<label>I <input id="te-I" type="range" min="0" max="8" step="0.1" value="3"><span id="te-Ival">3.0 A</span></label></div>
<div class="meters"><span>Qc: <b id="te-qc">–</b></span><span>COP: <b id="te-cop">–</b></span>
<span>ZT: <b id="te-zt">–</b></span><span id="te-oow"></span></div>
{chart}
{_svg_open(720, 140)}
<text x="16" y="24" fill="{_FG}" font-size="13" font-weight="bold">Segmented-leg steady T(x) at 3 A (Bi2Te3, frozen te_1d)</text>
{leg}
<text x="56" y="134" fill="{_AX}" font-size="10">cold → hot · uniform-leg limit reproduces the lumped model (parity below)</text>
</svg>
<h2>Pair table (per-pair S/R/K with contacts)</h2>
<table><tr><th>pair</th><th>S [µV/K]</th><th>R [mΩ]</th><th>K [mW/K]</th><th>Th max [°C]</th><th>I_opt [A]</th></tr>
{pairs_rows}</table>
<p class="dim">Parity: uniform-leg COP {d['parity']['leg_COP']:.4f} vs lumped COP {d['parity']['lumped_COP']:.4f} at 3 A. Validity windows are first-class: pairs driven past Th_max report out-of-window.</p>"""
    script = f"""const PAIRS = {json.dumps({n: {"S": c["S_uV_K"] * 1e-6, "R": c["R_mohm"] * 1e-3, "K": c["K_mW_K"] * 1e-3, "Thmax": c["Th_max_C"]} for n, c in d["curves"].items()})};
const TC = {d['Tc_C']}, TH = {d['Th_C']};
const LEG = {json.dumps(prof["T_C"])};
function teModel(S, R, K, I) {{
  const Tc = TC + 273.15, dT = TH - TC;
  const qc = S * Tc * I - 0.5 * I * I * R - K * dT;
  const w = S * I * dT + I * I * R;
  return {{ qc: Math.max(qc, 0), cop: (qc > 0 && w > 1e-12) ? qc / w : 0 }};
}}
function teUpdate() {{
  const name = document.getElementById("te-pair").value;
  const P = PAIRS[name], I = parseFloat(document.getElementById("te-I").value);
  document.getElementById("te-Ival").textContent = I.toFixed(1) + " A";
  const m = teModel(P.S, P.R, P.K, I);
  document.getElementById("te-qc").textContent = m.qc.toFixed(3) + " W";
  document.getElementById("te-cop").textContent = m.cop.toFixed(3);
  document.getElementById("te-zt").textContent = (P.S * P.S * ((TC + TH) / 2 + 273.15) / (P.R * P.K)).toFixed(2);
  const oow = TH > P.Thmax;
  document.getElementById("te-oow").textContent = oow ? "OUT-OF-WINDOW (past Th_max)" : "";
  const g = document.getElementById("te-x");
  g.innerHTML = "";
  const X = {x0} + ({x1} - {x0}) * I / 8;
  const mk = (tag, attrs) => {{ const el = document.createElementNS("http://www.w3.org/2000/svg", tag); for (const k in attrs) el.setAttribute(k, attrs[k]); g.appendChild(el); }};
  mk("line", {{ x1: X, y1: {y0}, x2: X, y2: {y1}, stroke: "#fff", "stroke-width": 1.5 }});
  const qmax = Math.max(...Object.values(PAIRS).map((p) => teModel(p.S, p.R, p.K, I).qc), 1e-9);
  const Y = {y1} - ({y1} - {y0}) * m.qc / (qmax * 1.05);
  mk("circle", {{ cx: X, cy: Y, r: 5, fill: "#fff" }});
}}
function teLeg() {{
  const lo = Math.min(...LEG), hi = Math.max(...LEG);
  LEG.forEach((v, k) => document.getElementById("te-leg-" + k).setAttribute("fill", inferno((v - lo) / Math.max(hi - lo, 1e-9))));
}}
function initTE() {{
  document.getElementById("te-pair").addEventListener("change", teUpdate);
  document.getElementById("te-I").addEventListener("input", teUpdate);
  teUpdate(); teLeg();
}}
document.addEventListener("DOMContentLoaded", initTE);"""
    return chrome("Thermoelectric pair + leg (frozen diffheat)", body, script)


def page_absorption(d: dict) -> str:
    x0, x1, y0, y1 = 56, 704, 40, 214
    tlo, thi = 60.0, 95.0
    qmax = max(max(c["COP"]) for c in d["curves"].values()) * 1.1
    cols = {"LiBr-H2O": _AMB, "NH3-H2O": _CYN, "LiCl-H2O": _GRN}
    series = []
    for name, c in d["curves"].items():
        px = [(_linmap(t, tlo, thi, x0, x1), _linmap(v, 0, qmax, y1, y0))
              for t, v in zip(c["t_gen"], c["COP"])]
        series.append({"px": px, "color": cols[name], "name": name})
    chart = curve_chart_svg("ab", series, "generator T [°C]", "COP")
    vessels = ""
    geom = [("gen", 90, "Generator"), ("con", 250, "Condenser"),
            ("eva", 410, "Evaporator"), ("abs", 570, "Absorber")]
    vg = [_svg_open(720, 200)]
    vg.append(f'<text x="16" y="24" fill="{_FG}" font-size="13" font-weight="bold">Single-effect machine — live values</text>')
    for key, cx, label in geom:
        vg.append(f'<rect x="{cx - 70}" y="50" width="140" height="100" rx="8" fill="{_PNL}" stroke="{_SPN}"/>')
        vg.append(f'<text x="{cx}" y="72" fill="{_FG}" font-size="11" text-anchor="middle">{label}</text>')
        vg.append(f'<text id="ab-{key}" x="{cx}" y="96" fill="{_AMB}" font-size="12" text-anchor="middle">–</text>')
        vg.append(f'<text id="ab-{key}2" x="{cx}" y="116" fill="{_AX}" font-size="10" text-anchor="middle">–</text>')
    for x1_, x2_ in ((160, 180), (320, 340), (480, 500)):
        vg.append(f'<line x1="{x1_}" y1="100" x2="{x2_}" y2="100" stroke="{_AX}" stroke-width="2"/>')
    vg.append(f'<line x1="640" y1="100" x2="660" y2="100" stroke="{_AX}" stroke-width="2"/>')
    vg.append(f'<line x1="60" y1="100" x2="20" y2="100" stroke="{_AX}" stroke-width="2"/>')
    vessels = "\n".join(vg) + "\n</svg>"
    pairs_rows = "\n".join(
        f"<tr><td>{n}</td><td>{c['eps']:.2f}</td><td>{c['dx_max']:.2f}</td></tr>"
        for n, c in d["curves"].items())
    body = f"""
<p class="dim"><a href="index.html">← all applications</a> · live tree core (harness.physics.absorption), generated {datetime.date.today().isoformat()}</p>
<h1>Absorption chiller — 4-vessel machine</h1>
<p class="dim">Ideal 3-temperature Novikov cycle derated by pair effectiveness, linear concentration-spread capacity. Move T_gen: machine values + chart dot recompute live. dx = 0 is the degenerate band bottom (no circulation → flows hide).</p>
<div class="math">COP_ideal = ((T_gen−T_cond)/T_gen)·(T_evap/(T_cond−T_evap)) · dx = clip(a·(T_gen−T_cond) − b·(T_cond−T_evap), 0, dx_max) · Q_evap = dx·h_fg · COP = eps·COP_ideal</div>
<div class="controls"><label>pair <select id="ab-pair"><option>LiBr-H2O</option><option>NH3-H2O</option><option>LiCl-H2O</option></select></label>
<label>T_gen <input id="ab-tgen" type="range" min="60" max="95" step="0.5" value="85"><span id="ab-tgenval">85 °C</span></label></div>
<div class="meters"><span>COP: <b id="ab-cop">–</b></span><span>SCP: <b id="ab-scp">–</b></span>
<span>dx: <b id="ab-dx">–</b></span><span id="ab-flag"></span></div>
{vessels}
{chart}
<h2>Working pairs</h2>
<table><tr><th>pair</th><th>effectiveness</th><th>dx_max</th></tr>{pairs_rows}</table>
<p class="dim">Calibration anchor: LiBr-H₂O at 7/30/85 °C gives COP {d['anchor']['COP']:.2f} (V-ABS gate ≈ 0.70 ±15%). NH₃-H₂O trades peak efficiency for a wider low-grade band.</p>"""
    script = f"""const PPR = {json.dumps({n: {"eps": c["eps"], "a": 0.00145 if "LiBr" in n else (0.002 if "NH3" in n else 0.0012), "b": 0.00085 if "LiBr" in n else (0.0006 if "NH3" in n else 0.0009), "dxmax": c["dx_max"]} for n, c in d["curves"].items()})};
const TE_C = {d['t_evap_c']}, TC_C = {d['t_cond_c']}, CYC = {d['cycle_time_s']};
function hfg(tK) {{ return 2256400.0 * Math.pow((647.1 - Math.min(tK, 646.1)) / (647.1 - 373.15), 0.321); }}
function abModel(pair, tgen) {{
  const P = PPR[pair], Te = TE_C + 273.15, Tc = TC_C + 273.15, Tg = tgen + 273.15;
  const ideal = (Tg > Tc && Tc > Te) ? ((Tg - Tc) / Tg) * (Te / Math.max(Tc - Te, 1e-6)) : 0;
  const dx = Math.min(Math.max(P.a * (tgen - TC_C) - P.b * (TC_C - TE_C), 0), P.dxmax);
  const qe = dx * hfg(Te);
  const cop = dx > 0 ? P.eps * ideal : 0;
  return {{ cop, scp: qe / CYC, dx, qe, qg: cop > 1e-12 ? qe / cop : 0, ideal }};
}}
function abUpdate() {{
  const pair = document.getElementById("ab-pair").value;
  const tg = parseFloat(document.getElementById("ab-tgen").value);
  document.getElementById("ab-tgenval").textContent = tg.toFixed(1) + " °C";
  const m = abModel(pair, tg);
  document.getElementById("ab-cop").textContent = m.cop.toFixed(3);
  document.getElementById("ab-scp").textContent = m.scp.toFixed(1) + " W/kg";
  document.getElementById("ab-dx").textContent = m.dx.toFixed(4);
  document.getElementById("ab-flag").textContent = m.dx <= 0 ? "DEGENERATE — no circulation" : "";
  document.getElementById("ab-gen").textContent = tg.toFixed(0) + " °C in";
  document.getElementById("ab-gen2").textContent = "Q " + (m.qg / 1000).toFixed(1) + " kJ/kg";
  document.getElementById("ab-con").textContent = TC_C.toFixed(0) + " °C";
  document.getElementById("ab-con2").textContent = "rejects heat";
  document.getElementById("ab-eva").textContent = TE_C.toFixed(0) + " °C";
  document.getElementById("ab-eva2").textContent = "Q " + (m.qe / 1000).toFixed(1) + " kJ/kg";
  document.getElementById("ab-abs").textContent = TC_C.toFixed(0) + " °C";
  document.getElementById("ab-abs2").textContent = m.dx > 0 ? "solution flows" : "no flow";
  const g = document.getElementById("ab-x");
  g.innerHTML = "";
  const X = {x0} + ({x1} - {x0}) * (tg - {tlo}) / ({thi} - {tlo});
  const Y = {y1} - ({y1} - {y0}) * m.cop / {qmax:.4f};
  const mk = (tag, attrs) => {{ const el = document.createElementNS("http://www.w3.org/2000/svg", tag); for (const k in attrs) el.setAttribute(k, attrs[k]); g.appendChild(el); }};
  mk("line", {{ x1: X, y1: {y0}, x2: X, y2: {y1}, stroke: "#fff", "stroke-width": 1.5 }});
  mk("circle", {{ cx: X, cy: Y, r: 5, fill: "#fff" }});
}}
function initAB() {{
  document.getElementById("ab-pair").addEventListener("change", abUpdate);
  document.getElementById("ab-tgen").addEventListener("input", abUpdate);
  abUpdate();
}}
document.addEventListener("DOMContentLoaded", initAB);"""
    return chrome("Absorption chiller (live tree core)", body, script)


_BUILDERS = {
    "pde-heat1d": (run_heat1d, page_heat1d),
    "pde-heat2d": (run_heat2d, page_heat2d),
    "pde-heat3d": (run_heat3d, page_heat3d),
    "pde-disc": (run_disc, page_disc),
    "pde-forced-conv": (run_forced, page_forced),
    "pde-boussinesq": (run_boussinesq, page_boussinesq),
    "pde-cloak": (run_cloak, page_cloak),
    "pde-telegrapher": (run_telegrapher, page_telegrapher),
    "pde-thermoelectric": (run_te, page_te),
    "pde-coldspot": (run_coldspot, page_coldspot),
    "pde-junction": (run_junction, page_junction),
    "pde-metamaterial": (run_metamaterial, page_metamaterial),
    "absorption": (run_absorption, page_absorption),
}


def write_pde_pages(out_dir: str | Path = "docs/apps",
                    which: list[str] | None = None,
                    quick: bool = False) -> dict[str, Path]:
    """Generate PDE application pages. Returns ``{slug: path}``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    slugs = list(which or [s for s, _, _ in PDE_PAGES])
    unknown = [s for s in slugs if s not in _BUILDERS]
    if unknown:
        raise KeyError(f"unknown PDE pages {unknown}; available: {sorted(_BUILDERS)}")
    written: dict[str, Path] = {}
    for slug in slugs:
        run, build = _BUILDERS[slug]
        if build is None:
            raise NotImplementedError(f"page builder for {slug!r} not yet written")
        t0 = time.time()
        data = run(quick=quick)
        path = out / f"{slug}.html"
        path.write_text(build(data), encoding="utf-8")
        written[slug] = path
        print(f"Wrote {path} ({slug}) {time.time() - t0:.1f}s")
    try:
        from .viz_apps import build_index  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        build_index = None
    if build_index is not None:
        build_index(out)
        written["index"] = out / "index.html"
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate frozen-PDE application pages.")
    ap.add_argument("--out", type=Path, default=Path("docs/apps"))
    ap.add_argument("--only", type=str, default="",
                    help="comma-separated slugs (default: all)")
    ap.add_argument("--quick", action="store_true",
                    help="small grids / short runs for previews and tests")
    args = ap.parse_args()
    which = [s.strip() for s in args.only.split(",") if s.strip()] or None
    t0 = time.time()
    write_pde_pages(args.out, which=which, quick=args.quick)
    print(f"{time.time() - t0:.1f}s total")


if __name__ == "__main__":
    main()

