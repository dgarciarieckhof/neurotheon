"""
turret_env.py – Phase-2½
─────────────────────────
Single-turret grid (21 × 21).  The turret is fixed at the centre and can fire
in eight compass directions (idle + ↑ ↓ ← → ↖ ↗ ↙ ↘).

Dynamics
• Enemies advance toward the turret every step   (enemy_move_every = 1).
• Allies drift in a slow random walk             (ally_move_every ≥ 1).
• Neutrals remain static but can be mis-classified as enemies with probability
  neutral_misclass_p when detected.
• Episode ends if (a) an enemy enters kill_radius, (b) all enemies destroyed,
  or (c) max_steps is reached.
• Safe-spawn: no entity spawns closer than safe_spawn_radius to the turret.
• Bullet shows for one frame; next frame an explosion tile appears, then clears.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .sensors import detect


class TurretEnv(gym.Env):
    metadata = {"render_modes": []}      # headless

    # ────────────────────────────────────────────────────────────────
    def __init__(
        self,
        grid_size: int = 21,
        max_steps: int = 120,
        sensor_range: int = 5,
        kill_radius: int = 1,
        safe_spawn_radius: int = 3,
        enemy_move_every: int = 1,
        ally_move_every: int = 3,
        neutral_misclass_p: float = 0.3,
        n_enemies: int = 7,
        n_allies: int = 4,
        n_neutrals: int = 3,
        enemy_stealth=(0.2, 0.8),
        ally_stealth: float = 0.1,
        neutral_stealth: float = 0.5,
        seed=None,
    ):
        super().__init__()
        # core parameters
        self.gs = grid_size
        self.max_steps = max_steps
        self.sensor_range = sensor_range
        self.kill_radius = kill_radius
        self.safe_spawn_radius = safe_spawn_radius
        self.enemy_move_every = enemy_move_every
        self.ally_move_every = ally_move_every
        self.neutral_misclass_p = neutral_misclass_p
        self.rng = np.random.default_rng(seed)

        # entity config
        self.cfg = dict(
            n_enemies=n_enemies,
            n_allies=n_allies,
            n_neutrals=n_neutrals,
            enemy_stealth=enemy_stealth,
            ally_stealth=ally_stealth,
            neutral_stealth=neutral_stealth,
        )

        # action space (idle + 8 compass directions)
        self.dir_vec = {
            0: np.array([0, 0]),    # idle
            1: np.array([-1, 0]),   # N
            2: np.array([1, 0]),    # S
            3: np.array([0, -1]),   # W
            4: np.array([0, 1]),    # E
            5: np.array([-1, -1]),  # NW
            6: np.array([-1, 1]),   # NE
            7: np.array([1, -1]),   # SW
            8: np.array([1, 1]),    # SE
        }
        self.action_space = spaces.Discrete(len(self.dir_vec))
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.gs, self.gs, 3), dtype=np.float32
        )

        # fixed turret position
        centre = self.gs // 2
        self.agent_pos = np.array([centre, centre], dtype=int)

        # runtime state vars
        self.entities = []
        self.bullets = []
        self.steps = 0

    # ─────────────────── internal helpers ──────────────────────────
    def _random_empty_cell(self):
        """Return a coordinate not at turret and outside safe radius."""
        while True:
            p = self.rng.integers(0, self.gs, 2)
            if (not np.array_equal(p, self.agent_pos) and
                np.linalg.norm(p - self.agent_pos) >= self.safe_spawn_radius):
                return p

    def _spawn_entities(self):
        ents = []
        for _ in range(self.cfg["n_enemies"]):
            ents.append(
                dict(
                    pos=self._random_empty_cell(),
                    type="enemy",
                    stealth=self.rng.uniform(*self.cfg["enemy_stealth"]),
                )
            )
        for _ in range(self.cfg["n_allies"]):
            ents.append(
                dict(
                    pos=self._random_empty_cell(),
                    type="ally",
                    stealth=self.cfg["ally_stealth"],
                )
            )
        for _ in range(self.cfg["n_neutrals"]):
            ents.append(
                dict(
                    pos=self._random_empty_cell(),
                    type="neutral",
                    stealth=self.cfg["neutral_stealth"],
                )
            )
        return ents

    def _get_obs(self):
        obs = np.zeros((self.gs, self.gs, 3), dtype=np.float32)
        for ent in self.entities:
            if detect(ent, self.agent_pos, self.sensor_range, self.rng):
                x, y = ent["pos"]
                if ent["type"] == "enemy":
                    obs[x, y, 0] = 1
                elif ent["type"] == "ally":
                    obs[x, y, 1] = 1
                else:  # neutral
                    if self.rng.random() < self.neutral_misclass_p:
                        obs[x, y, 0] = 1   # false red
        ax, ay = self.agent_pos
        obs[ax, ay, 2] = 1
        return obs

    def _move_entities(self):
        """Advance enemies toward turret; allies drift randomly."""
        if self.enemy_move_every and self.steps % self.enemy_move_every == 0:
            for ent in self.entities:
                if ent["type"] != "enemy":
                    continue
                direction = np.sign(self.agent_pos - ent["pos"])
                ent["pos"] = np.clip(ent["pos"] + direction, 0, self.gs - 1)

        if self.ally_move_every and self.steps % self.ally_move_every == 0:
            choices = list(self.dir_vec.values())[1:]  # exclude idle
            for ent in self.entities:
                if ent["type"] != "ally":
                    continue
                ent["pos"] = np.clip(
                    ent["pos"] + self.rng.choice(choices), 0, self.gs - 1
                )

    # ─────────────────── Gym API ───────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.entities = self._spawn_entities()
        self.bullets = []
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action: int):
        info = {
            "spawned_bullet": False,
            "hit": None,
            "explosion": None,
            "turret_destroyed": False,
        }
        reward = 0.0
        done = False

        # ─── spawn bullet
        if action != 0:
            self.bullets.append(
                dict(pos=self.agent_pos.copy(), dir=self.dir_vec[action])
            )
            info["spawned_bullet"] = True
        else:
            reward -= 0.02  # idle penalty

        # ─── advance bullets & check hits
        survivors = []
        for b in self.bullets:
            b["pos"] += b["dir"]
            x, y = b["pos"]
            if 0 <= x < self.gs and 0 <= y < self.gs:
                hit_idx = next(
                    (i for i, e in enumerate(self.entities)
                     if np.array_equal(e["pos"], b["pos"])),
                    None,
                )
                if hit_idx is not None:
                    ent = self.entities.pop(hit_idx)
                    info["explosion"] = b["pos"].copy()
                    if ent["type"] == "enemy":
                        reward += 1.0; info["hit"] = "enemy"
                    elif ent["type"] == "ally":
                        reward -= 2.0; info["hit"] = "ally"
                    else:
                        reward -= 0.5; info["hit"] = "neutral"
                    continue  # bullet disappears
                survivors.append(b)
        self.bullets = survivors

        # ─── move entities after bullet resolution
        self._move_entities()

        # ─── proximity defeat
        for ent in self.entities:
            if ent["type"] == "enemy" and \
               np.linalg.norm(ent["pos"] - self.agent_pos) <= self.kill_radius:
                done = True
                info["turret_destroyed"] = True
                reward -= 5.0
                break

        # ─── episode end
        self.steps += 1
        if self.steps >= self.max_steps or \
           not any(e["type"] == "enemy" for e in self.entities):
            done = True

        return self._get_obs(), reward, done, False, info

    def render(self):
        """No built-in rendering – Pygame viewer handles visuals."""
        pass
