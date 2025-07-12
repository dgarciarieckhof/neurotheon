"""
live_monitor.py
────────────────
Watch training progress in real time.

• Auto-loads newest checkpoint in runs/ every 20 s.
• Hot-swaps VecNormalize stats when they change.
• HUD shows FPS, step, reward, kills.
• Press TAB to toggle Grad-CAM heat-map overlay.

Start it any time—before, during, or after training.
"""
from __future__ import annotations
import glob, hashlib, os, time
import pygame, numpy as np

from envs.turret_env import TurretEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO

from sim.pygame_visual import draw, CELL
from sim.saliency import grad_cam

# ───── settings ──────────────────────────────────────────────────────
CHECK_EVERY = 20                       # seconds
VEC_PATH    = "runs/vecnorm.pkl"
CKPT_GLOB   = "runs/ppo_turret_step_*_steps.zip"

# ───── helpers ───────────────────────────────────────────────────────
def newest_ckpt() -> str | None:
    files = glob.glob(CKPT_GLOB)
    return max(files, key=os.path.getmtime) if files else None


def md5(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def load_vec_env() -> tuple[VecNormalize | DummyVecEnv, str | None]:
    """VecNormalize if stats exist, else DummyVecEnv."""
    base = DummyVecEnv([TurretEnv])
    if os.path.exists(VEC_PATH):
        env = VecNormalize.load(VEC_PATH, base)
        env.training = False
        env.norm_reward = False
        return env, md5(VEC_PATH)
    return base, None

# ───── main ──────────────────────────────────────────────────────────
def main() -> None:
    pygame.init()

    env, vec_hash = load_vec_env()
    gs = env.envs[0].gs
    win  = pygame.display.set_mode((gs * CELL, gs * CELL))
    font = pygame.font.SysFont(None, 20)
    clock = pygame.time.Clock()

    # Wait for first weights
    ckpt = newest_ckpt()
    while ckpt is None:
        print("Waiting for first checkpoint…")
        time.sleep(CHECK_EVERY)
        ckpt = newest_ckpt()

    print("🔄  Loading", ckpt)
    model = PPO.load(ckpt, env=env, device="cpu")

    obs = env.reset()[0]
    kills = step = score = 0
    cam_on, cam_cache = False, None
    expl_pos, ttl = None, 0
    last_check = time.time()

    running = True
    while running:
        # ─ hot-reload weights & vecnorm every CHECK_EVERY sec
        if time.time() - last_check > CHECK_EVERY:
            latest = newest_ckpt()
            if latest and latest != ckpt:
                print("🔄  Reloading", latest)
                model = PPO.load(latest, env=env, device="cpu")
                ckpt = latest

            new_hash = md5(VEC_PATH)
            if new_hash and new_hash != vec_hash:
                print("🔄  Reloading VecNormalize stats")
                env, vec_hash = load_vec_env()
                model.set_env(env)
            last_check = time.time()

        # ─ event handling
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                running = False
            if evt.type == pygame.KEYDOWN and evt.key == pygame.K_TAB:
                cam_on = not cam_on
                cam_cache = None            # invalidate overlay

        # ─ agent step
        action, _ = model.predict(obs, deterministic=True)
        action = int(action) if not isinstance(action, np.ndarray) else int(action.item())

        obs_b, r_b, done_b, info_b = env.step([action])
        obs, reward, done, info = obs_b[0], r_b[0], done_b[0], info_b[0]
        env.envs[0].observation = obs     # for draw()

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
            cam_cache = None

        # ─ draw grid + HUD
        draw(env.envs[0], win, font, clock.get_fps(),
             step, score, expl_pos, ttl, kills)

        # ─ Grad-CAM overlay
        if cam_on:
            if cam_cache is None:                       # compute once per frame
                heat = grad_cam(model, obs)             # 21×21 0-1
                overlay = pygame.Surface(win.get_size(), pygame.SRCALPHA)
                for x in range(gs):
                    for y in range(gs):
                        a = int(heat[x, y] * 120)
                        if a:
                            pygame.draw.rect(
                                overlay,
                                (255, 0, 0, a),
                                (y*CELL, x*CELL, CELL, CELL)
                            )
                cam_cache = overlay
            win.blit(cam_cache, (0, 0))
            pygame.display.flip()

        clock.tick(20)

    pygame.quit()


if __name__ == "__main__":
    main()