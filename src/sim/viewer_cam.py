"""
viewer_cam.py
Press TAB to toggle Grad-CAM overlay on/off.
"""
import pygame, numpy as np, sys
from envs.turret_env import TurretEnv
from stable_baselines3 import PPO
from sim.saliency import grad_cam
from sim.pygame_visual import draw, CELL
CAM_ALPHA = 120  # overlay transparency

def main(weights="runs/ppo_turret_final.zip"):
    pygame.init()
    env = TurretEnv()
    model = PPO.load(weights, device="cpu")

    win = pygame.display.set_mode((env.gs * CELL, env.gs * CELL))
    font = pygame.font.SysFont(None, 32)
    clock = pygame.time.Clock()

    obs, _ = env.reset()
    env.observation = obs
    overlay_on = False
    kills = score = step = 0
    expl_pos, ttl = None, 0

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT: pygame.quit(); sys.exit()
            if e.type == pygame.KEYDOWN and e.key == pygame.K_TAB:
                overlay_on = not overlay_on

        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, _, info = env.step(int(action))
        env.observation = obs
        score += r; step += 1
        if info.get("hit") == "enemy": kills += 1
        if info.get("explosion"): expl_pos, ttl = info["explosion"], 1
        elif ttl: ttl -= 1
        else: expl_pos = None
        if done:
            obs, _ = env.reset()
            env.observation = obs
            kills = score = step = 0

        # base grid
        draw(env, win, font, clock.get_fps(), step, score, expl_pos, ttl, kills)
        # overlay
        if overlay_on:
            heat = grad_cam(model, obs)  # 21×21 0-1
            surf = pygame.Surface(win.get_size(), pygame.SRCALPHA)
            for x in range(env.gs):
                for y in range(env.gs):
                    a = int(heat[x, y] * CAM_ALPHA)
                    if a:
                        rect = pygame.Rect(y*CELL, x*CELL, CELL, CELL)
                        surf.fill((255, 0, 0, a), rect)
            win.blit(surf, (0, 0))
            pygame.display.flip()
        clock.tick(20)

if __name__ == "__main__":
    main()
