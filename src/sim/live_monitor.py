"""
live_monitor.py
───────────────
Reload newest checkpoint from runs/ every 30 s and show an episode.
"""

from __future__ import annotations
import glob, os, time, pygame, numpy as np
from envs.turret_env import TurretEnv
from stable_baselines3 import PPO
from sim.pygame_visual import draw, CELL, MARGIN, COL  # reuse drawing helper

CHECK_EVERY = 20  # seconds

# --- checkpoint utils --------------------------------------------------
def newest_ckpt() -> str | None:
    files = glob.glob("runs/ppo_turret_step_*_steps.zip")
    return max(files, key=os.path.getmtime) if files else None

# --- main loop ---------------------------------------------------------
def main():
    pygame.init()
    env = TurretEnv()
    win = pygame.display.set_mode((env.gs * CELL, env.gs * CELL))
    font = pygame.font.SysFont(None, 20)
    clock = pygame.time.Clock()

    # wait for first checkpoint
    ckpt = newest_ckpt()
    while ckpt is None:
        print("Waiting for first checkpoint…")
        time.sleep(CHECK_EVERY)
        ckpt = newest_ckpt()

    print("🔄  Loading", ckpt)
    model = PPO.load(ckpt, env=env, device="cpu")

    obs, _ = env.reset()
    env.observation = obs
    expl_pos, ttl, step, score = None, 0, 0, 0.0
    last_check = time.time()

    running = True
    while running:
        # hot-reload weights
        if time.time() - last_check > CHECK_EVERY:
            latest = newest_ckpt()
            if latest and latest != ckpt:
                print("🔄  Reloading", latest)
                model = PPO.load(latest, env=env, device="cpu")
                ckpt = latest
            last_check = time.time()

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        # SB3 returns array shape (n_envs,) – squeeze to scalar
        action, _ = model.predict(obs, deterministic=True)
        if isinstance(action, np.ndarray):
            action = int(action.item())   # works for shape () or (1,)
        else:
            action = int(action)

        obs, r, done, _, info = env.step(action)
        env.observation = obs
        score += r
        step += 1

        if info.get("explosion") is not None:
            expl_pos, ttl = info["explosion"], 1
        elif ttl:
            ttl -= 1
        else:
            expl_pos = None

        if done:
            print(f"Episode finished | reward {score:.2f}")
            obs, _ = env.reset()
            env.observation = obs
            step = score = ttl = 0
            expl_pos = None

        draw(env, win, font, clock.get_fps(), step, score, expl_pos, ttl)
        clock.tick(20)

    pygame.quit()

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
