"""
Reload newest weights from runs/ every 30s and play an episode
so you can *see* learning progress while `train.py` is running.
"""
import glob, os, time, pygame
from envs.turret_env import TurretEnv
from stable_baselines3 import PPO
from sim.pygame_visual import draw, CELL, MARGIN, COL         # reuse routines

CHECK_EVERY = 30      # seconds

def newest_checkpoint():
    files = glob.glob("runs/ppo_turret_step-*.zip")
    return max(files, key=os.path.getmtime) if files else None

def main():
    pygame.init()
    env = TurretEnv()
    win = pygame.display.set_mode((env.gs*CELL, env.gs*CELL))
    font = pygame.font.SysFont(None, 20); clock = pygame.time.Clock()

    weights = newest_checkpoint()
    if not weights:
        print("Waiting for first checkpoint…")
        while not weights:
            time.sleep(CHECK_EVERY)
            weights = newest_checkpoint()

    model = PPO.load(weights, env=env, device="cpu")
    obs,_ = env.reset(); env.observation = obs
    explosion_pos=None; ttl=0; step=0; score=0

    t_last = time.time()
    running=True
    while running:
        # reload weights periodically
        if time.time()-t_last > CHECK_EVERY:
            latest = newest_checkpoint()
            if latest and latest != weights:
                print("🔄  Reloading", latest)
                model = PPO.load(latest, env=env, device="cpu")
                weights = latest
            t_last = time.time()

        for e in pygame.event.get():
            if e.type==pygame.QUIT: running=False
        action,_ = model.predict(obs, deterministic=True)
        obs,r,done,_,info = env.step(action)
        env.observation = obs
        score+=r; step+=1
        if info.get("explosion") is not None:
            explosion_pos=info["explosion"]; ttl=1
        elif ttl: ttl-=1
        else: explosion_pos=None
        if done:
            obs,_=env.reset(); env.observation=obs; step=score=0
        draw(env, win, font, clock.get_fps(), step, score,
             explosion_pos, ttl)
        clock.tick(20)
    pygame.quit()

if __name__=="__main__": main()