"""
Viewer for Phase-3 TurretEnv
Instant ray-cast (no bullet), one-frame explosion, slow movers.
"""
import sys, pygame
from envs.turret_env import TurretEnv

CELL,MARGIN,FPS=40,2,20
COL={"bg":(40,40,40),"und":(90,90,90),"tur":(25,25,250),
     "enm":(250,25,25),"ally":(25,250,25),"neu":(250,250,25),
     "exp":(255,140,0)}
KEY2ACT={pygame.K_UP:1,pygame.K_DOWN:2,pygame.K_LEFT:3,pygame.K_RIGHT:4,
         pygame.K_q:5,pygame.K_e:6,pygame.K_z:7,pygame.K_c:8,
         pygame.K_SPACE:0}

def draw(env,surf,font,fps,step,rew,expl_pos,ttl):
    gs=env.gs; surf.fill(COL["bg"])
    for x in range(gs):
        for y in range(gs):
            surf.fill(COL["und"],
                      pygame.Rect(y*CELL+MARGIN,x*CELL+MARGIN,CELL-2*MARGIN,CELL-2*MARGIN))
    # turret
    tx,ty=env.agent_pos
    pygame.draw.rect(surf,COL["tur"],
                     pygame.Rect(ty*CELL+MARGIN,tx*CELL+MARGIN,CELL-2*MARGIN,CELL-2*MARGIN))
    # entities (yellow for neutrals)
    for e in env.entities:
        x,y=e["pos"]; col={"enemy":"enm","ally":"ally","neutral":"neu"}[e["type"]]
        pygame.draw.rect(surf,COL[col],
                         pygame.Rect(y*CELL+MARGIN,x*CELL+MARGIN,CELL-2*MARGIN,CELL-2*MARGIN))
    # explosion
    if ttl and expl_pos is not None:
        ex,ey=expl_pos
        pygame.draw.rect(surf,COL["exp"],
                         pygame.Rect(ey*CELL+MARGIN,ex*CELL+MARGIN,CELL-2*MARGIN,CELL-2*MARGIN))
    hud=font.render(f"FPS {fps:.1f}  step {step}  reward {rew:.2f}",True,(255,255,255))
    surf.blit(hud,(5,5))
    pygame.display.flip()

def main():
    pygame.init(); env=TurretEnv()
    surf=pygame.display.set_mode((env.gs*CELL,env.gs*CELL))
    font=pygame.font.SysFont(None,20); clock=pygame.time.Clock()
    obs,_=env.reset(); env.observation=obs
    last_act,step,score=0,0,0.0; expl_pos=None; ttl=0
    run=True
    while run:
        for e in pygame.event.get():
            if e.type==pygame.QUIT: run=False
            elif e.type==pygame.KEYDOWN:
                if e.key in KEY2ACT: last_act=KEY2ACT[e.key]
                elif e.key==pygame.K_r:
                    obs,_=env.reset(); env.observation=obs
                    last_act=0;step=score=ttl=0;expl_pos=None
                elif e.key in (pygame.K_ESCAPE,pygame.K_x): run=False
        # env step
        obs,r,done,_,info=env.step(last_act); env.observation=obs
        score+=r; step+=1
        if info.get("explosion") is not None:
            expl_pos=info["explosion"]; ttl=1
        elif ttl: ttl-=1
        else: expl_pos=None
        if done:
            print("Episode finished reward",score)
            obs,_=env.reset(); env.observation=obs
            last_act=0;step=score=ttl=0;expl_pos=None
        draw(env,surf,font,clock.get_fps(),step,score,expl_pos,ttl)
        clock.tick(FPS)
    pygame.quit(); sys.exit()

if __name__=="__main__": main()
