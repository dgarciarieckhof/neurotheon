"""
live_monitor.py
────────────────
Continuously watch training progress:

• Auto-loads newest checkpoint in runs/ every 20 s.
• Uses matching VecNormalize stats when present (hot-swapped on update).
• Displays kills-per-episode, reward, FPS.

Start it any time—before, during, or after training.
"""

from __future__ import annotations
import glob, os, time, hashlib, pygame, numpy as np
from envs.turret_env import TurretEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO
from sim.pygame_visual import draw, CELL

CHECK_EVERY = 20            # seconds
VEC_PATH    = "runs/vecnorm.pkl"
CKPT_GLOB   = "runs/ppo_turret_step_*_steps.zip"


# ───────────────── helpers ──────────────────────────────────────────
def newest_ckpt() -> str | None:
    files = glob.glob(CKPT_GLOB)
    return max(files, key=os.path.getmtime) if files else None


def file_hash(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_vec_env() -> tuple[VecNormalize | DummyVecEnv, str | None]:
    """Return VecNormalize env (if stats exist) else raw DummyVecEnv."""
    base = DummyVecEnv([TurretEnv])
    if os.path.exists(VEC_PATH):
        env = VecNormalize.load(VEC_PATH, base)
        env.training = False        # evaluation mode
        env.norm_reward = False
        return env, file_hash(VEC_PATH)
    return base, None


# ─────────────────────────── main ───────────────────────────────────
def main() -> None:
    pygame.init()

    env, vec_hash = load_vec_env()

    win = pygame.display.set_mode(
        (env.envs[0].gs * CELL, env.envs[0].gs * CELL)
    )
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

    obs = env.reset()[0]          # unwrap batch dimension
    kills = step = score = 0
    expl_pos, ttl = None, 0
    last_check = time.time()

    running = True
    while running:
        # ── periodic hot-reload of weights & vecnorm
        if time.time() - last_check > CHECK_EVERY:
            latest = newest_ckpt()
            if latest and latest != ckpt:
                print("🔄  Reloading", latest)
                model = PPO.load(latest, env=env, device="cpu")
                ckpt = latest

            new_hash = file_hash(VEC_PATH)
            if new_hash and new_hash != vec_hash:
                print("🔄  Reloading VecNormalize stats")
                env, vec_hash = load_vec_env()
                model.set_env(env)

            last_check = time.time()

        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                running = False

        # predict action (ensure scalar)
        action, _ = model.predict(obs, deterministic=True)
        if isinstance(action, np.ndarray):
            action = int(action.item())
        else:
            action = int(action)

        obs_batch, r_batch, done_batch, info_batch = env.step([action])
        obs   = obs_batch[0]
        reward = r_batch[0]
        done   = done_batch[0]
        info   = info_batch[0]
        env.envs[0].observation = obs  # for draw()

        score += reward
        step  += 1
        if info.get("hit") == "enemy":
            kills += 1
        if info.get("explosion") is not None:
            expl_pos, ttl = info["explosion"], 1
        elif ttl:
            ttl -= 1
        else:
            expl_pos = None

        if done:
            print(f"Episode finished | reward {score:.2f} | kills {kills}")
            obs = env.reset()[0]
            kills = score = step = ttl = 0
            expl_pos = None

        draw(env.envs[0], win, font, clock.get_fps(),
             step, score, expl_pos, ttl, kills)
        clock.tick(20)

    pygame.quit()


if __name__ == "__main__":
    main()