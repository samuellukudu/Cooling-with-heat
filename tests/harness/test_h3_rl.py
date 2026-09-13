"""H3 RL backend — PPO behind harness[rl] extra, static wrapper, vectorized env.

Tests are lightweight: budget small, n_steps tiny, and the heavy PPO is mocked
so CI never needs torch/sb3. The fallback NotImplementedError when the extra
is missing is also verified.
"""

import sys
import types

import pytest

import harness
from harness.backends import RLBackend
from harness.envs.base import Objective
from harness.envs.cycle0d import build_cycle0d
from harness.envs.bed1d import Bed1D


def _objective_cop():
    return Objective.single("COP")


def test_rl_stub_raises_when_sb3_missing(monkeypatch):
    """Original honesty rule: missing extra -> NotImplementedError."""
    import importlib.util

    orig_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *a, **kw):
        if name in ("stable_baselines3", "torch"):
            return None
        return orig_find_spec(name, *a, **kw)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    backend = RLBackend()
    env = build_cycle0d("anchor:Silica gel RD", "datacenter")
    with pytest.raises(NotImplementedError, match="rl backend ships with the dynamic envs"):
        backend.solve(env, _objective_cop(), budget=32, seed=0)


def test_rl_uses_mocks_when_sb3_available(monkeypatch):
    """With a mocked SB3, the RL backend trains and returns OptimizeResult."""
    # Inject mocked sb3 and torch modules so find_spec succeeds
    # We create fake modules and patch find_spec to return non-None
    import importlib.util

    orig_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *a, **kw):
        if name.startswith("stable_baselines3") or name == "torch":
            return types.SimpleNamespace(name=name)  # truthy
        return orig_find_spec(name, *a, **kw)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    # Build fake sb3 modules
    fake_torch = types.ModuleType("torch")
    fake_sb3 = types.ModuleType("stable_baselines3")
    fake_vec = types.ModuleType("stable_baselines3.common.vec_env")
    fake_mon = types.ModuleType("stable_baselines3.common.monitor")

    class FakeMonitor:
        def __init__(self, env):
            self.env = env
            # forward attributes
            self.observation_space = env.observation_space
            self.action_space = env.action_space
        def reset(self, **kw):
            return self.env.reset(**kw)
        def step(self, action):
            return self.env.step(action)

    class FakeVecEnv:
        def __init__(self, fns):
            # fns is list of callables; call first
            self.env = fns[0]()
            self.observation_space = self.env.observation_space
            self.action_space = self.env.action_space
        def reset(self):
            obs, info = self.env.reset()
            # SB3 DummyVecEnv.reset returns obs only
            return obs
        def step(self, action):
            obs, rew, terminated, truncated, info = self.env.step(action)
            import numpy as np
            # VecEnv returns batched arrays + infos list
            return obs, np.array([rew]), np.array([terminated]), [info]

    fake_mon.Monitor = FakeMonitor
    fake_vec.DummyVecEnv = FakeVecEnv

    # PPO mock: does nothing in learn, predict returns sampled action
    class FakePPO:
        def __init__(self, policy, vec_env, verbose=0, seed=0, n_steps=32, **kw):
            self.vec_env = vec_env
            self.policy = policy
            self.seed = seed
            self.n_steps = n_steps
            self.learn_calls = []
        def learn(self, total_timesteps):
            self.learn_calls.append(total_timesteps)
        def predict(self, obs, deterministic=True):
            import numpy as np
            # Sample valid action from vec_env's env action_space
            space = self.vec_env.action_space
            if hasattr(space, "sample"):
                try:
                    # Need seeded sample; gymnasium spaces sample uses np random
                    act = space.sample()
                    # Ensure shape matches vectorized batch (1, dim)
                    # DummyVecEnv expects action batch; but FakeVecEnv steps with unbatched
                    # PPO predict returns action array
                    return np.asarray(act), None
                except Exception:
                    pass
            # fallback
            import numpy as np
            if hasattr(space, "low"):
                lo = np.asarray(space.low); hi = np.asarray(space.high)
                mid = (lo + hi) / 2.0
                return mid, None
            return np.array([0]), None

    fake_sb3.PPO = FakePPO

    # Inject into sys.modules for import_module
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "stable_baselines3", fake_sb3)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common.vec_env", fake_vec)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common.monitor", fake_mon)
    # also need import_module to find them; patch import_module fallback?
    # monkeypatch importlib.import_module to return our fakes when needed
    orig_import_module = __import__("importlib").import_module

    def fake_import_module(name, package=None):
        if name == "gymnasium":
            return orig_import_module(name, package)
        if name == "torch":
            return fake_torch
        if name == "stable_baselines3":
            return fake_sb3
        if name == "stable_baselines3.common.vec_env":
            return fake_vec
        if name == "stable_baselines3.common.monitor":
            return fake_mon
        return orig_import_module(name, package)

    monkeypatch.setattr("importlib.import_module", fake_import_module)

    # Test static wrapper (Cycle0D)
    env = build_cycle0d("anchor:Silica gel RD", "datacenter")
    backend = RLBackend()
    res = backend.solve(env, _objective_cop(), budget=64, seed=0, n_steps=16, n_eval_episodes=2, verbose=0)
    assert res.problem == "Cycle0D-v0"
    assert res.backend == "rl"
    assert isinstance(res.best_metrics, dict)
    assert "COP" in res.best_metrics
    assert isinstance(res.history, list) and len(res.history) >= 1
    assert res.n_evals >= 64
    assert res.best_objective == pytest.approx(res.best_metrics["COP"], rel=1e-9)  # objective single COP
    # best_design should be non-empty for static
    assert isinstance(res.best_design, dict) and len(res.best_design) > 0
    assert res.extra["is_static"] is True

    # Test dynamic wrapper (Bed1D) with mocked env
    bed = Bed1D(material="anchor:Silica gel RD", profile="datacenter", n_cells=4, n_cycles=1, dt_phys_s=0.02)
    res2 = backend.solve(bed, Objective.single("SCP_W_kg"), budget=48, seed=1, n_steps=16, n_eval_episodes=2, verbose=0)
    assert res2.problem == "Bed1D-v0"
    assert res2.backend == "rl"
    assert "SCP_W_kg" in res2.best_metrics or "COP" in res2.best_metrics
    assert isinstance(res2.history, list) and len(res2.history) >= 1
    assert res2.extra["is_static"] is False


def test_rl_via_harness_optimize_with_mock(monkeypatch):
    """harness.optimize(backend='rl') dispatches through registry and uses mocked PPO."""
    import importlib.util, types, sys
    orig_find_spec = importlib.util.find_spec
    def fake_find_spec(name, *a, **kw):
        if name.startswith("stable_baselines3") or name == "torch":
            return types.SimpleNamespace(name=name)
        return orig_find_spec(name, *a, **kw)
    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    fake_torch = types.ModuleType("torch")
    fake_sb3 = types.ModuleType("stable_baselines3")
    fake_vec = types.ModuleType("stable_baselines3.common.vec_env")
    fake_mon = types.ModuleType("stable_baselines3.common.monitor")
    class FakeMonitor:
        def __init__(self, env):
            self.env=env; self.observation_space=env.observation_space; self.action_space=env.action_space
        def reset(self, **kw): return self.env.reset(**kw)
        def step(self, a): return self.env.step(a)
    class FakeVecEnv:
        def __init__(self, fns):
            self.env=fns[0](); self.observation_space=self.env.observation_space; self.action_space=self.env.action_space
        def reset(self):
            o,_ = self.env.reset(); return o
        def step(self, a):
            import numpy as np
            o,r,d,_,info=self.env.step(a); return o, np.array([r]), np.array([d]), [info]
    fake_mon.Monitor=FakeMonitor; fake_vec.DummyVecEnv=FakeVecEnv
    class FakePPO:
        def __init__(self, policy, vec_env, verbose=0, seed=0, n_steps=32, **kw): self.vec_env=vec_env
        def learn(self, total_timesteps): pass
        def predict(self, obs, deterministic=True):
            import numpy as np
            sp = self.vec_env.action_space
            if hasattr(sp, "sample"):
                try:
                    return np.asarray(sp.sample()), None
                except: pass
            if hasattr(sp, "low"):
                return (np.asarray(sp.low)+np.asarray(sp.high))/2.0, None
            return np.array([0]), None
    fake_sb3.PPO=FakePPO
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "stable_baselines3", fake_sb3)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common.vec_env", fake_vec)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common.monitor", fake_mon)
    orig_import_module = __import__("importlib").import_module
    def fake_import_module(name, package=None):
        if name in ("torch","stable_baselines3","stable_baselines3.common.vec_env","stable_baselines3.common.monitor"):
            return {"torch":fake_torch,"stable_baselines3":fake_sb3,"stable_baselines3.common.vec_env":fake_vec,"stable_baselines3.common.monitor":fake_mon}[name]
        if name=="gymnasium": return orig_import_module(name,package)
        return orig_import_module(name,package)
    monkeypatch.setattr("importlib.import_module", fake_import_module)
    env = build_cycle0d("anchor:Silica gel RD", "datacenter")
    res = harness.optimize(env, Objective.single("COP"), backend="rl", seed=0, budget=32, n_steps=16, n_eval_episodes=1)
    assert res.backend == "rl"
    assert res.history
