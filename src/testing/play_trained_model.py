from __future__ import annotations
import glob, hashlib, os, time, pathlib
import pygame, numpy as np

from envs.turret_env import TurretEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO

# reuse colours, CELL size, shake constant and draw() from the viewer
from sim.pygame_visual import draw, CELL, SHAKE_MS, SND_EXPLO
from sim.saliency import grad_cam

# ─── settings ────────────────────────────────────────────────────────
RUNS_DIR    = pathlib.Path(__file__).resolve().parents[2] / "runs"
MODEL_PATH  = RUNS_DIR / "best_model" / "best_model.zip"
VEC_PATH    = RUNS_DIR / "vecnorm.pkl"
CKPT_GLOB   = str(RUNS_DIR / "ppo_turret_step_*_steps.zip")
CHECK_EVERY = 20  # seconds

# ─── helpers ─────────────────────────────────────────────────────────
def newest_ckpt() -> str | None:
    files = glob.glob(CKPT_GLOB)
    return max(files, key=os.path.getmtime) if files else None


def md5(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def load_vec_env() -> tuple[VecNormalize | DummyVecEnv, str | None]:
    base = DummyVecEnv([TurretEnv])
    if os.path.exists(VEC_PATH):
        env = VecNormalize.load(VEC_PATH, base)
        env.training = False
        env.norm_reward = False
        return env, md5(str(VEC_PATH))
    return base, None

# ─── main ────────────────────────────────────────────────────────────
def main() -> None:
    pygame.init()

    env, vec_hash = load_vec_env()
    gs  = env.envs[0].gs
    win = pygame.display.set_mode((gs * CELL, gs * CELL))
    font = pygame.font.SysFont(None, 32)
    clock = pygame.time.Clock()

    # fallback if no checkpoints found
    ckpt = newest_ckpt()
    if ckpt is None:
        if MODEL_PATH.exists():
            print("📦  Loading static model:", MODEL_PATH)
            model = PPO.load(str(MODEL_PATH), env=env, device="cpu")
        else:
            raise FileNotFoundError("No checkpoint or best_model.zip found.")
    else:
        print("🔄  Loading", ckpt)
        model = PPO.load(ckpt, env=env, device="cpu")

    obs   = env.reset()[0]
    kills = step = score = 0

    explosions: list[dict] = []
    shake_off = 0

    cam_on, cam_cache = False, None

    last_check = time.time()
    running = True
    while running:
        if time.time() - last_check > CHECK_EVERY:
            latest = newest_ckpt()
            if latest and latest != ckpt:
                try:
                    print("🔄  Reloading", latest)
                    model = PPO.load(latest, env=env, device="cpu")
                    ckpt = latest
                except Exception as e:
                    print(f"⚠️  Failed to load {latest} — {e}")

            new_hash = md5(str(VEC_PATH))
            if new_hash and new_hash != vec_hash:
                print("🔄  Reloading VecNormalize stats")
                env, vec_hash = load_vec_env()
                model.set_env(env)
            last_check = time.time()

        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                running = False
            elif evt.type == pygame.KEYDOWN and evt.key == pygame.K_TAB:
                cam_on  = not cam_on
                cam_cache = None

        action, _ = model.predict(obs, deterministic=True)
        action = int(action) if not isinstance(action, np.ndarray) else int(action.item())

        obs_b, r_b, done_b, info_b = env.step([action])
        obs, reward, done, info = obs_b[0], r_b[0], done_b[0], info_b[0]
        env.envs[0].observation = obs

        score += reward
        step  += 1
        if info.get("hit") == "enemy":
            kills += 1

        exp = info.get("explosion", None)
        if exp is not None:
            explosions.append(dict(pos=tuple(exp), frame=0))
            shake_off = SHAKE_MS // (1000 // 20)
            if SND_EXPLO:
                SND_EXPLO.play()

        if shake_off:
            shake_off -= 1

        if done:
            print(f"Episode finished | reward {score:.2f} | kills {kills}")
            obs = env.reset()[0]
            kills = score = step = 0
            explosions.clear()
            shake_off   = 0
            cam_cache   = None

        time_left = env.envs[0].max_steps - step
        shake_off = draw(
            env.envs[0], win, font, clock.get_fps(),
            step, score, kills, time_left,
            explosions, shake_off,
            mission_banner=None
        )

        if cam_on:
            if cam_cache is None:
                heat = grad_cam(model, obs)
                overlay = pygame.Surface(win.get_size(), pygame.SRCALPHA)
                for x in range(gs):
                    for y in range(gs):
                        a = int(heat[x, y] * 120)
                        if a:
                            pygame.draw.rect(
                                overlay, (255, 0, 0, a),
                                (y*CELL, x*CELL, CELL, CELL)
                            )
                cam_cache = overlay
            win.blit(cam_cache, (0, 0))
            pygame.display.flip()

        clock.tick(20)

    pygame.quit()


if __name__ == "__main__":
    main()
