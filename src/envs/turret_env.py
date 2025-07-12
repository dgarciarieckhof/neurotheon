"""
turret_env.py
───────────────────────
• Fixed turret at grid centre (21 × 21 board)
• Instant 8-direction ray-cast (idle + ↑ ↓ ← → ↖ ↗ ↙ ↘)
• Enemies advance toward turret every `enemy_move_every` steps
• Allies drift randomly every `ally_move_every` steps
• Neutrals are static yellow blocks
• Safe spawn: no entity within `safe_spawn_radius`
• Explosion coordinate returned in info dict for one-frame flash
• Enemy pathing modes: direct / zig-zag / spiral.
• Turret fire-rate cooldown (cannot spam shots).
• Dynamic fog-of-war: sensor range shrinks, then resets.
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .sensors import detect


class TurretEnv(gym.Env):
    metadata = {"render_modes": []}

    # ────────────────────────────────────────────────────────────────
    def __init__(
        self,
        grid_size: int = 21,
        max_steps: int = 300,
        sensor_range: int = 6,        # initial range
        sensor_min: int = 3,          # min range when fog is thick
        fog_shrink_every: int = 25,   # steps before range–1 (then reset)
        kill_radius: int = 1,
        safe_spawn_radius: int = 3,

        enemy_move_every: int = 5,
        ally_move_every: int = 10,

        cooldown_steps: int = 1,      # turret cooldown

        path_probs: tuple[float, float, float] = (0.6, 0.25, 0.15),
        zigzag_period: int = 4,
        spiral_radius: int = 4,

        n_enemies: int = 7,
        n_allies: int = 4,
        n_neutrals: int = 3,
        enemy_stealth=(0.2, 0.8),
        ally_stealth: float = 0.1,
        neutral_stealth: float = 0.0,
        seed=None,
    ):
        super().__init__()

        # core params
        self.gs = grid_size
        self.max_steps = max_steps

        self.sensor_range_init = sensor_range
        self.sensor_range = sensor_range
        self.sensor_min = sensor_min
        self.fog_shrink_every = fog_shrink_every

        self.kill_radius = kill_radius
        self.safe_spawn_radius = safe_spawn_radius

        self.enemy_move_every = enemy_move_every
        self.ally_move_every = ally_move_every

        self.cooldown_steps = cooldown_steps
        self.last_shot_step = -cooldown_steps

        self.path_probs = path_probs
        self.zigzag_period = zigzag_period
        self.spiral_radius = spiral_radius

        self.rng = np.random.default_rng(seed)

        self.cfg = dict(
            n_enemies=n_enemies,
            n_allies=n_allies,
            n_neutrals=n_neutrals,
            enemy_stealth=enemy_stealth,
            ally_stealth=ally_stealth,
            neutral_stealth=neutral_stealth,
        )

        # actions: idle + 8 ray-cast directions
        self.vec = {
            0: np.array([0, 0]),
            1: np.array([-1, 0]), 2: np.array([1, 0]),
            3: np.array([0, -1]), 4: np.array([0, 1]),
            5: np.array([-1, -1]), 6: np.array([-1, 1]),
            7: np.array([1, -1]), 8: np.array([1, 1]),
        }
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            0, 1, shape=(self.gs, self.gs, 3), dtype=np.float32
        )

        centre = self.gs // 2
        self.agent_pos = np.array([centre, centre])
        self.reset()

    # ───────────────────────── helpers ──────────────────────────────
    def _random_cell(self, used: set[tuple[int, int]]) -> np.ndarray:
        while True:
            p = tuple(self.rng.integers(0, self.gs, 2))
            if p in used:
                continue
            if np.linalg.norm(np.array(p) - self.agent_pos) >= self.safe_spawn_radius:
                used.add(p)
                return np.array(p)

    def _spawn_entities(self):
        used = {tuple(self.agent_pos)}
        ents = []

        # ─ enemies with pathing mode
        for _ in range(self.cfg["n_enemies"]):
            mode = self.rng.choice(
                ["direct", "zigzag", "spiral"],
                p=self.path_probs
            )
            ents.append(dict(
                pos=self._random_cell(used),
                type="enemy",
                mode=mode,
                zig_state=0,
                spiral_angle=self.rng.random() * 2 * np.pi,
                stealth=self.rng.uniform(*self.cfg["enemy_stealth"]),
            ))

        # ─ allies
        for _ in range(self.cfg["n_allies"]):
            ents.append(dict(
                pos=self._random_cell(used),
                type="ally",
                stealth=self.cfg["ally_stealth"],
            ))

        # ─ neutrals (static)
        for _ in range(self.cfg["n_neutrals"]):
            ents.append(dict(
                pos=self._random_cell(used),
                type="neutral",
                stealth=self.cfg["neutral_stealth"],
            ))
        return ents

    def _get_obs(self):
        obs = np.zeros((self.gs, self.gs, 3), np.float32)
        for e in self.entities:
            if detect(e, self.agent_pos, self.sensor_range, self.rng):
                x, y = e["pos"]
                if e["type"] == "enemy":
                    obs[x, y, 0] = 1
                elif e["type"] == "ally":
                    obs[x, y, 1] = 1
                else:  # neutral
                    obs[x, y, 0] = obs[x, y, 1] = 1
        ax, ay = self.agent_pos
        obs[ax, ay, 2] = 1
        return obs

    # ───────────────────── movement logic ──────────────────────────
    def _move(self):
        # enemies
        if self.steps % self.enemy_move_every == 0:
            for e in self.entities:
                if e["type"] != "enemy":
                    continue
                if e["mode"] == "direct":
                    e["pos"] += np.sign(self.agent_pos - e["pos"])

                elif e["mode"] == "zigzag":
                    main = np.sign(self.agent_pos - e["pos"])
                    perp = main[::-1] * np.array([1, -1])  # rotate 90°
                    if e["zig_state"] // self.zigzag_period % 2 == 0:
                        step = main
                    else:
                        step = perp
                    e["zig_state"] += 1
                    e["pos"] += step

                else:  # spiral
                    # rotate step around centre, gradually tighten
                    radial = np.sign(self.agent_pos - e["pos"])
                    tang = np.array([-radial[1], radial[0]])
                    e["spiral_angle"] += np.pi / self.spiral_radius
                    step = np.round(
                        np.cos(e["spiral_angle"]) * radial +
                        np.sin(e["spiral_angle"]) * tang
                    ).astype(int)
                    e["pos"] += step

                e["pos"] = np.clip(e["pos"], 0, self.gs - 1)

        # allies drift
        if self.steps % self.ally_move_every == 0:
            choices = list(self.vec.values())[1:]
            for e in self.entities:
                if e["type"] != "ally":
                    continue
                e["pos"] = np.clip(
                    e["pos"] + self.rng.choice(choices),
                    0, self.gs - 1
                )

    # ───────────────────── Gym API ─────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.entities = self._spawn_entities()
        self.steps = 0
        self.sensor_range = self.sensor_range_init
        self.last_shot_step = -self.cooldown_steps
        return self._get_obs(), {}

    def step(self, action: int):
        info = {"hit": None, "explosion": None, "turret_destroyed": False}
        reward, done = 0.0, False

        # ─── shooting with cooldown
        can_fire = (self.steps - self.last_shot_step) >= self.cooldown_steps
        if action != 0 and can_fire:
            self.last_shot_step = self.steps
            pos = self.agent_pos.copy()
            while True:
                pos = pos + self.vec[action]
                x, y = pos
                if not (0 <= x < self.gs and 0 <= y < self.gs):
                    break
                idx = next((i for i, e in enumerate(self.entities)
                            if np.array_equal(e["pos"], pos)), None)
                if idx is not None:
                    e = self.entities.pop(idx)
                    info["explosion"] = pos.copy()
                    if e["type"] == "enemy":
                        reward += 1.0; info["hit"] = "enemy"
                    elif e["type"] == "ally":
                        reward -= 1.0; info["hit"] = "ally"
                    else:
                        reward -= 0.4; info["hit"] = "neutral"
                    break

        elif action != 0 and not can_fire:
            reward -= 0.05   # tried to shoot too early
        else:
            reward -= 0.01   # idle

        # ─── move entities, fog-of-war update
        self._move()

        # proximity defeat
        for e in self.entities:
            if e["type"] == "enemy" and \
               np.linalg.norm(e["pos"] - self.agent_pos) <= self.kill_radius:
                done = True
                info["turret_destroyed"] = True
                reward -= 5.0
                break

        # sensor range breathing
        if self.fog_shrink_every and self.steps % self.fog_shrink_every == 0:
            if self.sensor_range > self.sensor_min:
                self.sensor_range -= 1
            else:
                self.sensor_range = self.sensor_range_init

        # episode end?
        self.steps += 1
        if self.steps >= self.max_steps or \
           not any(e["type"] == "enemy" for e in self.entities):
            done = True

        return self._get_obs(), reward, done, False, info

    def render(self):  # viewer handles drawing
        pass