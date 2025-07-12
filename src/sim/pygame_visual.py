# sim/pygame_visual.py  – polished night-theme viewer
import sys, pygame, pathlib, math
from envs.turret_env import TurretEnv

# ───────── assets & constants ──────────────────────────────────────
CELL, MARGIN, FPS = 40, 2, 20
SHAKE_MS, SHAKE_PIX = 50, 4

ASSET_DIR   = pathlib.Path(__file__).resolve().parents[2] / "assets"
EXP_SHEET   = pygame.image.load(ASSET_DIR / "explosion.png")
EXP_FRAMES  = [EXP_SHEET.subsurface(pygame.Rect(col*32, row*32, 32, 32))
               for row in range(4) for col in range(4)]
EXP_FRAMES = [pygame.transform.smoothscale(f, (CELL, CELL))
              for f in EXP_FRAMES]

# ---- initialise mixer only if possible ----------------------------
try:
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    SND_EXPLO = pygame.mixer.Sound(ASSET_DIR / "explosion.mp3")
except pygame.error:
    SND_EXPLO = None          # head-less ⇒ just skip sound

COL = dict(
    bg   =(0, 0, 0),          # black background
    grid =(255, 255, 255),    # white grid lines   ← NEW
    und  =(25, 25, 25),       # very dark cell fill
    sh   =(0, 0, 0, 80),      # soft shadow alpha
    tur  =( 50, 120, 255),    # bright blue
    enm  =(255,  60,  60),    # vivid red
    ally =(  0, 255, 150),    # teal-green
    neu  =(255, 255,  60),    # bright yellow
)

KEY2ACT = {pygame.K_UP:1,pygame.K_DOWN:2,pygame.K_LEFT:3,pygame.K_RIGHT:4,
           pygame.K_q:5,pygame.K_e:6,pygame.K_z:7,pygame.K_c:8,
           pygame.K_SPACE:0}

# ───────── helper ─────────────────────────────────────────────────
def _shadow_rect(x, y):
    return pygame.Rect(y*CELL+MARGIN, x*CELL+MARGIN,
                       CELL-2*MARGIN, CELL-2*MARGIN)

# ───────── draw  ──────────────────────────────────────────────────
def draw(env, surf, font, fps, step, score, kills, time_left,
         explosions, shake_off, mission_banner):

    gs = env.gs
    surf.fill(COL["bg"])

    # camera shake
    if shake_off:
        off_x = int(math.sin(shake_off*31) * SHAKE_PIX)
        off_y = int(math.cos(shake_off*17) * SHAKE_PIX)
        shake_off -= 1
    else:
        off_x = off_y = 0

    # cells + shadows
    grid_surf = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    for x in range(gs):
        for y in range(gs):
            pygame.draw.rect(grid_surf, COL["und"], _shadow_rect(x, y))
            pygame.draw.rect(grid_surf, COL["sh"],  _shadow_rect(x, y))
    surf.blit(grid_surf, (off_x, off_y))

    # white grid lines  ← NEW
    for i in range(gs + 1):
        x = i * CELL + off_x
        y = i * CELL + off_y
        pygame.draw.line(surf, COL["grid"], (x, 0), (x, gs*CELL), 1)
        pygame.draw.line(surf, COL["grid"], (0, y), (gs*CELL, y), 1)

    # entities
    for e in env.entities:
        x, y = e["pos"]
        col  = COL[{"enemy":"enm","ally":"ally","neutral":"neu"}[e["type"]]]
        pygame.draw.rect(surf, col, _shadow_rect(x, y).move(off_x, off_y))

    # turret
    tx, ty = env.agent_pos
    pygame.draw.rect(surf, COL["tur"], _shadow_rect(tx, ty).move(off_x, off_y))

    # explosion animation
    for ex in explosions[:]:
        if ex["frame"] >= len(EXP_FRAMES):
            explosions.remove(ex); continue
        img = EXP_FRAMES[ex["frame"]]
        px, py = ex["pos"]
        surf.blit(img, (py*CELL + off_x, px*CELL + off_y))
        ex["frame"] += 1

    # HUD
    hud = font.render(
        f"FPS {fps:>4.1f}   step {step}/{env.max_steps}   "
        f"reward {score:6.2f}   kills {kills}   T-{time_left}",
        True, (255, 255, 255)
    )
    surf.blit(hud, (6, 4))

    # mission banner (currently disabled)  ← REMOVED / COMMENTED
    # if mission_banner:
    #     banner = font.render(mission_banner, True, (255, 255, 0))
    #     bx = (surf.get_width() - banner.get_width()) // 2
    #     surf.blit(banner, (bx, surf.get_height()//2 - 10))

    pygame.display.flip()
    return shake_off

# ───────── main viewer ────────────────────────────────────────────
def main():
    pygame.init(); pygame.mixer.init()
    env  = TurretEnv()
    surf = pygame.display.set_mode((env.gs*CELL, env.gs*CELL))
    font = pygame.font.SysFont(None, 32)
    clock= pygame.time.Clock()

    obs, _      = env.reset(); env.observation = obs
    last_act    = 0
    step = score = kills = 0
    explosions  = []
    shake_off   = 0
    # banner      = None  # banner logic disabled  ← REMOVED

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT: running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in KEY2ACT: last_act = KEY2ACT[e.key]
                elif e.key == pygame.K_r:
                    obs, _ = env.reset(); env.observation = obs
                    last_act = step = score = kills = 0
                    explosions.clear()
                    # banner = None
                elif e.key in (pygame.K_ESCAPE, pygame.K_x):
                    running = False

        obs, r, done, _, info = env.step(last_act); env.observation = obs
        step += 1; score += r
        if info.get("hit") == "enemy": kills += 1
        if info.get("explosion"):
            explosions.append(dict(pos=info["explosion"], frame=0))
            shake_off = SHAKE_MS // (1000 // FPS)
            if SND_EXPLO: 
                SND_EXPLO.play()

        shake_off = draw(
            env, surf, font, clock.get_fps(),
            step, score, kills,
            env.max_steps - step,
            explosions, shake_off,
            mission_banner=None          # always None now
        )
        clock.tick(FPS)

        if done:
            # episode reset (no banner)
            obs, _ = env.reset(); env.observation = obs
            last_act = step = score = kills = 0
            explosions.clear()

    pygame.quit(); sys.exit()


if __name__ == "__main__":
    main()