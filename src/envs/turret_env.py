"""
turret_env.py – Phase-3
───────────────────────
* Enemies move one cell toward the turret every `enemy_move_every` steps.
* Allies drift one random cell every `ally_move_every` steps.
* Neutrals are static yellow blocks.
* Firing is instantaneous (ray-cast). Success shows explosion for renderer.
"""

import numpy as np, gymnasium as gym
from gymnasium import spaces
from .sensors import detect


class TurretEnv(gym.Env):
    metadata = {"render_modes": []}

    # ─────────────────────────────────────────────
    def __init__(
        self,
        grid_size: int = 21,
        max_steps: int = 150,
        sensor_range: int = 6,
        kill_radius: int = 1,
        safe_spawn_radius: int = 3,
        enemy_move_every: int = 5,   # ← slower enemies
        ally_move_every: int = 7,    # ← even slower allies
        n_enemies: int = 7,
        n_allies: int = 4,
        n_neutrals: int = 3,
        enemy_stealth=(0.2, 0.8),
        ally_stealth: float = 0.1,
        neutral_stealth: float = 0.0,   # neutrals always visible
        neutral_misclass_p: float = 0.0, # turned off – they’re their own colour
        seed=None,
    ):
        super().__init__()
        self.gs = grid_size
        self.max_steps = max_steps
        self.sensor_range = sensor_range
        self.kill_radius = kill_radius
        self.safe_spawn_radius = safe_spawn_radius
        self.enemy_move_every = enemy_move_every
        self.ally_move_every = ally_move_every
        self.rng = np.random.default_rng(seed)

        self.cfg = dict(
            n_enemies=n_enemies, n_allies=n_allies, n_neutrals=n_neutrals,
            enemy_stealth=enemy_stealth, ally_stealth=ally_stealth,
            neutral_stealth=neutral_stealth,
        )

        # idle + 8 compass directions
        self.vec = {
            0: np.array([0, 0]),
            1: np.array([-1, 0]), 2: np.array([1, 0]),
            3: np.array([0,-1]), 4: np.array([0, 1]),
            5: np.array([-1,-1]),6: np.array([-1, 1]),
            7: np.array([ 1,-1]),8: np.array([ 1, 1]),
        }
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            0, 1, shape=(self.gs, self.gs, 3), dtype=np.float32)

        centre = self.gs // 2
        self.agent_pos = np.array([centre, centre])
        self.reset()

    # ───────────────── spawn / helpers ───────────
    def _random_cell(self, used):
        while True:
            p = tuple(self.rng.integers(0, self.gs, 2))
            if p in used: continue
            if np.linalg.norm(np.array(p) - self.agent_pos) >= self.safe_spawn_radius:
                used.add(p); return np.array(p)

    def _spawn_entities(self):
        used = {tuple(self.agent_pos)}
        ents=[]
        for _ in range(self.cfg["n_enemies"]):
            ents.append(dict(pos=self._random_cell(used),type="enemy",
                             stealth=self.rng.uniform(*self.cfg["enemy_stealth"])))
        for _ in range(self.cfg["n_allies"]):
            ents.append(dict(pos=self._random_cell(used),type="ally",
                             stealth=self.cfg["ally_stealth"]))
        for _ in range(self.cfg["n_neutrals"]):
            ents.append(dict(pos=self._random_cell(used),type="neutral",
                             stealth=self.cfg["neutral_stealth"]))
        return ents

    def _get_obs(self):
        obs = np.zeros((self.gs, self.gs, 3), np.float32)
        for e in self.entities:
            if detect(e, self.agent_pos, self.sensor_range, self.rng):
                x,y = e["pos"]
                if   e["type"]=="enemy":   obs[x,y,0]=1
                elif e["type"]=="ally":    obs[x,y,1]=1
                else:  # neutral
                    obs[x,y,0]=1; obs[x,y,1]=1  # yellow
        ax,ay=self.agent_pos; obs[ax,ay,2]=1
        return obs

    # ───────────────── movement ──────────────────
    def _move(self):
        if self.steps % self.enemy_move_every == 0:
            for e in self.entities:
                if e["type"]!="enemy": continue
                e["pos"]=np.clip(e["pos"]+np.sign(self.agent_pos-e["pos"]),0,self.gs-1)
        if self.steps % self.ally_move_every == 0:
            dirs=list(self.vec.values())[1:]
            for e in self.entities:
                if e["type"]!="ally": continue
                e["pos"]=np.clip(e["pos"]+self.rng.choice(dirs),0,self.gs-1)

    # ───────────────── Gym API ───────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.entities=self._spawn_entities()
        self.steps=0
        return self._get_obs(),{}

    def step(self, action:int):
        info={"hit":None,"explosion":None,"turret_destroyed":False}
        reward,done=0.0,False

        # ─── instant laser / ray-cast ───
        if action:
            pos = self.agent_pos.copy()
            while True:
                pos = pos + self.vec[action]
                x,y = pos
                if not (0<=x<self.gs and 0<=y<self.gs): break
                hit_idx=next((i for i,e in enumerate(self.entities)
                              if np.array_equal(e["pos"],pos)),None)
                if hit_idx is not None:
                    e=self.entities.pop(hit_idx)
                    info["explosion"]=pos.copy()
                    if   e["type"]=="enemy": reward+=1;   info["hit"]="enemy"
                    elif e["type"]=="ally":  reward-=2;   info["hit"]="ally"
                    else:                    reward-=0.5; info["hit"]="neutral"
                    break
        else:
            reward-=0.02  # idle cost

        # move entities after shot
        self._move()

        # proximity defeat
        for e in self.entities:
            if e["type"]=="enemy" and np.linalg.norm(e["pos"]-self.agent_pos)<=self.kill_radius:
                done=True; info["turret_destroyed"]=True; reward-=5; break

        self.steps+=1
        if self.steps>=self.max_steps or not any(e["type"]=="enemy" for e in self.entities):
            done=True
        return self._get_obs(),reward,done,False,info

    def render(self): pass