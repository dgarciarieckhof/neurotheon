from types import SimpleNamespace
from agents.curriculum import CurriculumCallback
from envs.turret_env import TurretEnv


def make_env(**kw):
    return TurretEnv(**kw)


def test_curriculum_switch():
    cb = CurriculumCallback(
        make_env,
        milestones=[(0, {"n_enemies": 1}), (3, {"n_enemies": 5})],
    )

    dummy_vec = SimpleNamespace(envs=[make_env(n_enemies=1)])
    fake_model = SimpleNamespace(get_env=lambda: dummy_vec)
    cb.model = fake_model  # SB3 attaches this

    cb.num_timesteps = 0
    cb._on_step()
    assert dummy_vec.envs[0].cfg["n_enemies"] == 1

    cb.num_timesteps = 3
    cb._on_step()
    assert dummy_vec.envs[0].cfg["n_enemies"] == 5