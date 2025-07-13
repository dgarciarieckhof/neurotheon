"""
turret_env.py - Enhanced with visual effect support
───────────────────────────────────────────────────
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
• Enhanced visual support: shot trajectories, entity IDs, last action tracking
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .sensors import detect


class TurretEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        grid_size: int = 31,
        max_steps: int = 300,
        sensor_range: int = 10,
        sensor_min: int = 4,
        fog_shrink_every: int = 15,
        kill_radius: int = 1,
        safe_spawn_radius: int = 3,
        enemy_move_every: int = 5,
        ally_move_every: int = 10,
        cooldown_steps: int = 1,
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
        
        # Enhanced visual support attributes
        self.last_action = 0
        self.shot_trajectory = None  # For visual effects
        self.entity_id_counter = 0   # For tracking entity movement
        self.previous_entities = []  # For movement tracking
        
        self.reset()

    def _is_valid_position(self, pos: np.ndarray) -> bool:
        """Check if position is within grid bounds"""
        return 0 <= pos[0] < self.gs and 0 <= pos[1] < self.gs

    def _is_safe_spawn(self, pos: np.ndarray, used: set[tuple[int, int]]) -> bool:
        """Check if position is safe for spawning (not too close to turret or other entities)"""
        if tuple(pos) in used:
            return False
        if np.linalg.norm(pos - self.agent_pos) < self.safe_spawn_radius:
            return False
        # Check minimum spacing between entities
        for existing_pos in used:
            if np.linalg.norm(pos - np.array(existing_pos)) < 2:
                return False
        return True

    def _random_cell(self, used: set[tuple[int, int]]) -> np.ndarray:
        max_attempts = 100
        for _ in range(max_attempts):
            p = self.rng.integers(0, self.gs, 2)
            if self._is_safe_spawn(p, used):
                used.add(tuple(p))
                return p
        # Fallback: find any valid position
        for x in range(self.gs):
            for y in range(self.gs):
                p = np.array([x, y])
                if self._is_safe_spawn(p, used):
                    used.add(tuple(p))
                    return p
        raise RuntimeError("Cannot find valid spawn position")

    def _spawn_entities(self):
        used = {tuple(self.agent_pos)}
        ents = []

        for _ in range(self.cfg["n_enemies"]):
            mode = self.rng.choice(["direct", "zigzag", "spiral"], p=self.path_probs)
            entity = dict(
                pos=self._random_cell(used),
                type="enemy",
                mode=mode,
                zig_state=0,
                spiral_angle=self.rng.random() * 2 * np.pi,
                stealth=self.rng.uniform(*self.cfg["enemy_stealth"]),
                entity_id=self.entity_id_counter,  # For tracking
                previous_pos=None,  # For movement tracking
            )
            self.entity_id_counter += 1
            ents.append(entity)

        for _ in range(self.cfg["n_allies"]):
            entity = dict(
                pos=self._random_cell(used),
                type="ally",
                stealth=self.cfg["ally_stealth"],
                entity_id=self.entity_id_counter,
                previous_pos=None,
            )
            self.entity_id_counter += 1
            ents.append(entity)

        for _ in range(self.cfg["n_neutrals"]):
            entity = dict(
                pos=self._random_cell(used),
                type="neutral",
                stealth=self.cfg["neutral_stealth"],
                entity_id=self.entity_id_counter,
                previous_pos=None,
            )
            self.entity_id_counter += 1
            ents.append(entity)

        return ents

    def _get_obs(self):
        obs = np.zeros((self.gs, self.gs, 3), np.float32)
        for e in self.entities:
            if detect(e, self.agent_pos, self.sensor_range, self.rng):
                x, y = e["pos"]
                # Add bounds checking for observation
                if 0 <= x < self.gs and 0 <= y < self.gs:
                    if e["type"] == "enemy":
                        obs[x, y, 0] = 1
                    elif e["type"] == "ally":
                        obs[x, y, 1] = 1
                    else:  # neutral
                        obs[x, y, 0] = obs[x, y, 1] = 1
        ax, ay = self.agent_pos
        obs[ax, ay, 2] = 1
        return obs

    def _calculate_shot_trajectory(self, action: int) -> list[np.ndarray]:
        """Calculate the complete trajectory of a shot for visual effects"""
        if action == 0:
            return []
        
        trajectory = []
        pos = self.agent_pos.copy()
        direction = self.vec[action]
        
        while True:
            pos = pos + direction
            if not self._is_valid_position(pos):
                break
            trajectory.append(pos.copy())
            
            # Check if we hit an entity
            if any(np.array_equal(e["pos"], pos) for e in self.entities):
                break
        
        return trajectory

    def _move(self):
        # Store previous positions for movement tracking
        for e in self.entities:
            e["previous_pos"] = e["pos"].copy()
        
        if self.steps % self.enemy_move_every == 0:
            for e in self.entities:
                if e["type"] != "enemy":
                    continue
                
                if e["mode"] == "direct":
                    step = np.sign(self.agent_pos - e["pos"])
                elif e["mode"] == "zigzag":
                    main = np.sign(self.agent_pos - e["pos"])
                    perp = main[::-1] * np.array([1, -1])
                    # Improved zigzag calculation
                    if e["zig_state"] % (self.zigzag_period * 2) < self.zigzag_period:
                        step = main
                    else:
                        step = perp
                    e["zig_state"] += 1
                else:  # spiral
                    radial = np.sign(self.agent_pos - e["pos"])
                    tang = np.array([-radial[1], radial[0]])
                    # Wrap angle to prevent overflow
                    e["spiral_angle"] = (e["spiral_angle"] + np.pi / self.spiral_radius) % (2 * np.pi)
                    step = np.round(
                        np.cos(e["spiral_angle"]) * radial +
                        np.sin(e["spiral_angle"]) * tang
                    ).astype(int)
                
                # Better bounds checking instead of clipping
                new_pos = e["pos"] + step
                if self._is_valid_position(new_pos):
                    e["pos"] = new_pos
                else:
                    # Try alternative moves if blocked
                    for alt_step in [np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1])]:
                        alt_pos = e["pos"] + alt_step
                        if self._is_valid_position(alt_pos):
                            e["pos"] = alt_pos
                            break

        if self.steps % self.ally_move_every == 0:
            choices = list(self.vec.values())[1:]
            for e in self.entities:
                if e["type"] != "ally":
                    continue
                step = self.rng.choice(choices)
                new_pos = e["pos"] + step
                if self._is_valid_position(new_pos):
                    e["pos"] = new_pos

    def can_fire(self) -> bool:
        """Check if turret can fire (not on cooldown)"""
        return (self.steps - self.last_shot_step) >= self.cooldown_steps

    def get_nearest_enemy(self) -> dict | None:
        """Get the nearest enemy entity for turret targeting"""
        nearest = None
        min_dist = float('inf')
        
        for e in self.entities:
            if e["type"] == "enemy":
                dist = np.linalg.norm(e["pos"] - self.agent_pos)
                if dist < min_dist:
                    min_dist = dist
                    nearest = e
        
        return nearest

    def get_enemies_in_range(self, radius: float) -> list[dict]:
        """Get all enemies within specified radius of turret"""
        enemies_in_range = []
        for e in self.entities:
            if e["type"] == "enemy":
                dist = np.linalg.norm(e["pos"] - self.agent_pos)
                if dist <= radius:
                    enemies_in_range.append(e)
        return enemies_in_range

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.entities = self._spawn_entities()
        self.steps = 0
        self.sensor_range = self.sensor_range_init
        self.last_shot_step = -self.cooldown_steps
        self.turret_hits = 0
        self.ally_hit_count = 0
        
        # Reset visual tracking
        self.last_action = 0
        self.shot_trajectory = None
        self.previous_entities = []
        
        return self._get_obs(), {}

    def step(self, action: int):
        info = {"hit": None, "explosion": None, "turret_destroyed": False}
        reward, done = 0.0, False
        
        # Store the action for visual effects
        self.last_action = action
        
        # Calculate shot trajectory for visual effects
        self.shot_trajectory = self._calculate_shot_trajectory(action)

        can_fire = self.can_fire()
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
                    info["shot_trajectory"] = self.shot_trajectory  # For visual effects
                    if e["type"] == "enemy":
                        reward += 2.0 + 0.01 * (self.max_steps - self.steps)
                        info["hit"] = "enemy"
                    elif e["type"] == "ally":
                        self.ally_hit_count += 1
                        penalty = 1.0 + 0.2 * self.ally_hit_count
                        reward -= penalty
                        info["hit"] = "ally"
                    else:
                        reward -= 0.3
                        info["hit"] = "neutral"
                    break
        elif action != 0 and not can_fire:
            reward -= 0.02
            info["cooldown_violation"] = True  # For visual feedback
        elif action == 0:
            # Smaller penalty for strategic waiting
            reward -= 0.001
        else:
            reward -= 0.005

        # Store entity states before movement for tracking
        self.previous_entities = [e.copy() for e in self.entities]
        
        self._move()

        # turret hit handling - FIXED
        enemy_hits = [
            e for e in self.entities
            if e["type"] == "enemy" and
            np.linalg.norm(e["pos"] - self.agent_pos) <= self.kill_radius
        ]

        if enemy_hits:
            self.turret_hits += len(enemy_hits)
            info["turret_hit"] = len(enemy_hits)  # For visual effects
            
            # Remove enemies that hit the turret
            self.entities = [
                e for e in self.entities
                if not (e["type"] == "enemy" and 
                        np.linalg.norm(e["pos"] - self.agent_pos) <= self.kill_radius)
            ]
            
            if self.turret_hits >= 3:
                reward -= 3.0
                done = True
                info["turret_destroyed"] = True
            else:
                reward -= 0.5 * len(enemy_hits) + 0.01 * self.steps

        # Enhanced fog logic with gradual recovery
        if self.fog_shrink_every and self.steps % self.fog_shrink_every == 0:
            old_range = self.sensor_range
            if self.sensor_range > self.sensor_min:
                self.sensor_range -= 1
            elif self.steps % (self.fog_shrink_every * 2) == 0:
                # Gradual recovery instead of instant reset
                self.sensor_range = min(self.sensor_range + 1, self.sensor_range_init)
            
            if old_range != self.sensor_range:
                info["sensor_range_changed"] = (old_range, self.sensor_range)

        self.steps += 1

        # Enhanced info for visual effects
        info["enemies_remaining"] = len([e for e in self.entities if e["type"] == "enemy"])
        info["allies_remaining"] = len([e for e in self.entities if e["type"] == "ally"])
        info["can_fire"] = self.can_fire()
        info["sensor_range"] = self.sensor_range
        info["nearest_enemy"] = self.get_nearest_enemy()
        info["enemies_in_danger_zone"] = self.get_enemies_in_range(5.0)

        # final end check only if not done already
        if not done:
            if not any(e["type"] == "enemy" for e in self.entities):
                reward += 5.0
                done = True
                info["victory"] = True
            elif self.steps >= self.max_steps:
                done = True
                info["timeout"] = True

        return self._get_obs(), reward, done, False, info

    def render(self):
        pass