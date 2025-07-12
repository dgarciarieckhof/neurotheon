from envs.turret_env import TurretEnv

def test_basic_loop():
    env = TurretEnv()
    obs, _ = env.reset()
    assert obs.shape == (env.gs, env.gs, 3)

    done, ep_reward = False, 0
    steps = 0
    while not done and steps < 20:
        action = env.action_space.sample()
        obs, r, done, _, info = env.step(action)
        ep_reward += r
        steps += 1

    # At least one step should modify reward either positively or negatively
    assert ep_reward != 0
    env.close()