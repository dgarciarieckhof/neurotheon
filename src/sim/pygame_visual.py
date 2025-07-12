"""
pygame_visual.py
────────────────
Manual viewer for TurretEnv (21×21, 8-dir fire, explosions).

Controls
  ↑ ↓ ← →   shoot N S W E
  q e z c   shoot NW NE SW SE
  SPACE     idle
  R         reset episode
  ESC / X   quit
"""
from __future__ import annotations
import sys, pygame
from envs.turret_env import TurretEnv

# ────────── visuals
CELL   = 40                 # pixel size of a grid cell (21×21 → 840 px window)
MARGIN = 2
FPS    = 30
COL = {
    "bg":  (40, 40, 40),
    "tur": ( 25,  25,250),
    "enm": (250,  25, 25),
    "ally":( 25, 250, 25),
    "neu": (250, 250, 25),
    "bul": (255, 200,  0),
    "exp": (255, 140,  0),
    "und": ( 90,  90, 90),
}
KEY2ACT = {
    pygame.K_UP:    1, pygame.K_DOWN:  2,
    pygame.K_LEFT:  3, pygame.K_RIGHT: 4,
    pygame.K_q:     5, pygame.K_e:     6,
    pygame.K_z:     7, pygame.K_c:     8,
    pygame.K_SPACE: 0,
}
LEGEND = "↑↓←→, q e z c = fire · R reset · Esc/X quit"

# ────────── draw helper
def draw(env: TurretEnv, surf, font, fps: float, step: int,
         reward: float, explosion_pos, explosion_ttl):
    gs = env.gs
    surf.fill(COL["bg"])

    # background grid
    for x in range(gs):
        for y in range(gs):
            rect = pygame.Rect(y*CELL+MARGIN, x*CELL+MARGIN,
                               CELL-2*MARGIN, CELL-2*MARGIN)
            surf.fill(COL["und"], rect)

    # turret FIRST so bullet overlays
    tx, ty = env.agent_pos
    pygame.draw.rect(surf, COL["tur"],
                     pygame.Rect(ty*CELL+MARGIN, tx*CELL+MARGIN,
                                 CELL-2*MARGIN, CELL-2*MARGIN))

    # entities
    for e in env.entities:
        x, y = e["pos"]
        if env.observation[x, y, 0]:
            col = COL["enm"]
        elif env.observation[x, y, 1]:
            col = COL["ally"]
        else:
            continue       # neutrals invisible unless misclassified
        pygame.draw.rect(surf, col,
                         pygame.Rect(y*CELL+MARGIN, x*CELL+MARGIN,
                                     CELL-2*MARGIN, CELL-2*MARGIN))

    # bullets
    for b in env.bullets:
        bx, by = b["pos"]
        center = (by*CELL+CELL//2, bx*CELL+CELL//2)
        pygame.draw.circle(surf, COL["bul"], center, CELL//6)

    # explosion (one frame after hit)
    if explosion_ttl and explosion_pos is not None:
        ex, ey = explosion_pos
        e_rect = pygame.Rect(ey*CELL+MARGIN, ex*CELL+MARGIN,
                             CELL-2*MARGIN, CELL-2*MARGIN)
        pygame.draw.rect(surf, COL["exp"], e_rect)

    # HUD
    txt = font.render(f"FPS {fps:.1f} | step {step} | reward {reward:.2f}",
                      True, (255,255,255))
    surf.blit(txt, (5,5))
    surf.blit(font.render(LEGEND, True, (200,200,200)), (5,25))

    pygame.display.flip()

# ────────── main loop
def main():
    pygame.init()
    env  = TurretEnv()
    size = env.gs * CELL
    surf = pygame.display.set_mode((size, size))
    font = pygame.font.SysFont(None, 20)
    clock= pygame.time.Clock()

    obs,_ = env.reset()
    env.observation = obs
    last_action     = 0
    step, score     = 0, 0.0
    explosion_pos, explosion_ttl = None, 0

    running=True
    while running:
        # ── input
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running=False
            elif e.type == pygame.KEYDOWN:
                if e.key in KEY2ACT:
                    last_action = KEY2ACT[e.key]
                elif e.key == pygame.K_r:
                    obs,_ = env.reset()
                    env.observation = obs
                    last_action = 0; step = 0; score = 0.0
                elif e.key in (pygame.K_ESCAPE, pygame.K_x):
                    running=False

        # ── env step every frame
        obs, r, done, _, info = env.step(last_action)
        env.observation = obs
        score += r; step += 1

        # explosion bookkeeping (one extra frame)
        if info.get("explosion") is not None:
            explosion_pos = info["explosion"]
            explosion_ttl = 1
        elif explosion_ttl:
            explosion_ttl -= 1
        else:
            explosion_pos = None

        if done:
            print("Episode finished | reward:", score)
            obs,_ = env.reset()
            env.observation = obs
            last_action = 0; step = 0; score = 0.0
            explosion_pos = None; explosion_ttl = 0

        draw(env, surf, font, clock.get_fps(), step, score,
             explosion_pos, explosion_ttl)
        clock.tick(FPS)

    pygame.quit(); sys.exit()

# ────────── entry
if __name__ == "__main__":
    main()
