import numpy as np
from envs.turret_env import TurretEnv


def test_enemy_moves_exactly_one_cell():
    env = TurretEnv(seed=1, n_enemies=0, enemy_move_every=4)

    start = env.agent_pos + np.array([-4, -4])
    env.entities.append(dict(pos=start.copy(), type="enemy", stealth=0.0))
    idx = len(env.entities) - 1           # remember our enemy

    k = env.enemy_move_every
    for _ in range(k):
        env.step(0)

    moved = env.entities[idx]["pos"]
    assert np.sum(np.abs(moved - start)) == 2      # (±1, ±1)