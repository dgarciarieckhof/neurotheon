"""
turret_env.py - Enhanced with dual shooting capability
─────────────────────────────────────────────────────
• Fixed turret at grid centre (21 × 21 board)
• Dual-shot system: can shoot two enemies simultaneously in different directions
• 8-direction ray-cast (idle + ↑ ↓ ← → ↖ ↗ ↙ ↘)
• Enhanced action space for dual shooting combinations
• Enemies advance toward turret every `enemy_move_every` steps
• Allies drift randomly every `ally_move_every` steps
• Neutrals are static yellow blocks
• Safe spawn: no entity within `safe_spawn_radius`
• Explosion coordinates returned in info dict for visual effects
• Enemy pathing modes: direct / zig-zag / spiral
• Turret fire-rate cooldown (cannot spam shots)
• Dynamic fog-of-war: sensor range shrinks, then resets
• Enhanced visual support: shot trajectories, entity IDs, last action tracking
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .sensors import detect
from itertools import combinations


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
        dual_shot_cooldown: int = 3,  # Higher cooldown for dual shots
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
        self.dual_shot_cooldown = dual_shot_cooldown
        self.last_shot_step = -cooldown_steps
        self.last_dual_shot_step = -dual_shot_cooldown
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

        # Create action mapping for dual shots
        self._create_action_mapping()

        self.observation_space = spaces.Box(
            0, 1, shape=(self.gs, self.gs, 3), dtype=np.float32
        )

        centre = self.gs // 2
        self.agent_pos = np.array([centre, centre])
        
        # Enhanced visual support attributes
        self.last_action = 0
        self.shot_trajectories = []  # Multiple trajectories for dual shots
        self.entity_id_counter = 0   # For tracking entity movement
        self.previous_entities = []  # For movement tracking
        
        self.reset()

    def _create_action_mapping(self):
        """Create action mapping for single shots, dual shots, and idle"""
        self.action_mapping = {}
        action_idx = 0
        
        # Action 0: Idle (no shot)
        self.action_mapping[action_idx] = {"type": "idle", "directions": []}
        action_idx += 1
        
        # Actions 1-8: Single shots in 8 directions
        for direction in range(1, 9):
            self.action_mapping[action_idx] = {"type": "single", "directions": [direction]}
            action_idx += 1
        
        # Actions 9-36: Dual shots (combinations of 2 different directions)
        for dir1, dir2 in combinations(range(1, 9), 2):
            self.action_mapping[action_idx] = {"type": "dual", "directions": [dir1, dir2]}
            action_idx += 1
        
        self.action_space = spaces.Discrete(action_idx)
        
        # Create reverse mapping for convenience
        self.reverse_action_mapping = {v["type"]: [] for v in self.action_mapping.values()}
        for action_idx, action_info in self.action_mapping.items():
            self.reverse_action_mapping[action_info["type"]].append(action_idx)

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

    def _calculate_shot_trajectory(self, direction: int) -> list[np.ndarray]:
        """Calculate the complete trajectory of a shot for visual effects"""
        if direction == 0:
            return []
        
        trajectory = []
        pos = self.agent_pos.copy()
        direction_vec = self.vec[direction]
        
        while True:
            pos = pos + direction_vec
            if not self._is_valid_position(pos):
                break
            trajectory.append(pos.copy())
            
            # Check if we hit an entity
            if any(np.array_equal(e["pos"], pos) for e in self.entities):
                break
        
        return trajectory

    def _process_shot(self, direction: int) -> tuple[float, dict]:
        """Process a single shot in the given direction"""
        reward = 0.0
        shot_info = {"hit": None, "explosion": None, "trajectory": []}
        
        trajectory = self._calculate_shot_trajectory(direction)
        shot_info["trajectory"] = trajectory
        
        pos = self.agent_pos.copy()
        while True:
            pos = pos + self.vec[direction]
            x, y = pos
            if not (0 <= x < self.gs and 0 <= y < self.gs):
                break
            
            idx = next((i for i, e in enumerate(self.entities)
                       if np.array_equal(e["pos"], pos)), None)
            if idx is not None:
                e = self.entities.pop(idx)
                shot_info["explosion"] = pos.copy()
                
                if e["type"] == "enemy":
                    reward += 3.0 + 0.01 * (self.max_steps - self.steps)
                    shot_info["hit"] = "enemy"
                elif e["type"] == "ally":
                    self.ally_hit_count += 1
                    penalty = 1.0 + 0.2 * self.ally_hit_count
                    reward -= penalty
                    shot_info["hit"] = "ally"
                else:
                    reward -= 0.5
                    shot_info["hit"] = "neutral"
                break
        
        return reward, shot_info

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

    def can_fire(self, shot_type: str = "single") -> bool:
        """Check if turret can fire (not on cooldown)"""
        if shot_type == "dual":
            return (self.steps - self.last_dual_shot_step) >= self.dual_shot_cooldown
        else:
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

    def get_action_description(self, action: int) -> str:
        """Get human-readable description of action"""
        if action not in self.action_mapping:
            return f"Invalid action: {action}"
        
        action_info = self.action_mapping[action]
        if action_info["type"] == "idle":
            return "Idle (no shot)"
        elif action_info["type"] == "single":
            direction_names = ["", "N", "S", "W", "E", "NW", "NE", "SW", "SE"]
            dir_name = direction_names[action_info["directions"][0]]
            return f"Single shot {dir_name}"
        else:  # dual
            direction_names = ["", "N", "S", "W", "E", "NW", "NE", "SW", "SE"]
            dir1_name = direction_names[action_info["directions"][0]]
            dir2_name = direction_names[action_info["directions"][1]]
            return f"Dual shot {dir1_name} + {dir2_name}"

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.entities = self._spawn_entities()
        self.steps = 0
        self.sensor_range = self.sensor_range_init
        self.last_shot_step = -self.cooldown_steps
        self.last_dual_shot_step = -self.dual_shot_cooldown
        self.turret_hits = 0
        self.ally_hit_count = 0
        
        # Reset visual tracking
        self.last_action = 0
        self.shot_trajectories = []
        self.previous_entities = []
        
        return self._get_obs(), {}

    def step(self, action: int):
        info = {"hit": None, "explosion": None, "explosions": [], "turret_destroyed": False, "shots_fired": []}
        reward, done = 0.0, False
        
        # Store the action for visual effects
        self.last_action = action
        
        # Get action info
        if action not in self.action_mapping:
            raise ValueError(f"Invalid action: {action}")
        
        action_info = self.action_mapping[action]
        self.shot_trajectories = []
        
        # Process shooting
        if action_info["type"] == "idle":
            # Small penalty for strategic waiting
            reward -= 0.001
            
        elif action_info["type"] == "single":
            if self.can_fire("single"):
                self.last_shot_step = self.steps
                direction = action_info["directions"][0]
                shot_reward, shot_info = self._process_shot(direction)
                reward += shot_reward
                
                self.shot_trajectories.append(shot_info["trajectory"])
                info["shots_fired"].append({"direction": direction, "type": "single"})
                
                if shot_info["hit"]:
                    info["hit"] = shot_info["hit"]
                if shot_info["explosion"] is not None:
                    info["explosion"] = shot_info["explosion"]  # For backward compatibility
                    info["explosions"].append(shot_info["explosion"])
            else:
                reward -= 0.02
                info["cooldown_violation"] = True
                
        elif action_info["type"] == "dual":
            if self.can_fire("dual"):
                self.last_dual_shot_step = self.steps
                self.last_shot_step = self.steps  # Also update single shot cooldown
                
                total_hits = []
                for direction in action_info["directions"]:
                    shot_reward, shot_info = self._process_shot(direction)
                    reward += shot_reward
                    
                    self.shot_trajectories.append(shot_info["trajectory"])
                    info["shots_fired"].append({"direction": direction, "type": "dual"})
                    
                    if shot_info["hit"]:
                        total_hits.append(shot_info["hit"])
                    if shot_info["explosion"] is not None:
                        info["explosions"].append(shot_info["explosion"])
                        # Set the first explosion as the main explosion for backward compatibility
                        if info["explosion"] is None:
                            info["explosion"] = shot_info["explosion"]
                
                # Bonus for dual shot efficiency
                if len(total_hits) >= 2:
                    reward += 0.5  # Bonus for hitting multiple targets
                
                info["hit"] = total_hits if total_hits else None
            else:
                reward -= 0.05  # Higher penalty for dual shot cooldown violation
                info["dual_cooldown_violation"] = True

        # Store entity states before movement for tracking
        self.previous_entities = [e.copy() for e in self.entities]
        
        self._move()

        # Turret hit handling
        enemy_hits = [
            e for e in self.entities
            if e["type"] == "enemy" and
            np.linalg.norm(e["pos"] - self.agent_pos) <= self.kill_radius
        ]

        if enemy_hits:
            self.turret_hits += len(enemy_hits)
            info["turret_hit"] = len(enemy_hits)
            
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
        info["can_fire_single"] = self.can_fire("single")
        info["can_fire_dual"] = self.can_fire("dual")
        info["sensor_range"] = self.sensor_range
        info["nearest_enemy"] = self.get_nearest_enemy()
        info["enemies_in_danger_zone"] = self.get_enemies_in_range(5.0)
        info["shot_trajectories"] = self.shot_trajectories
        info["action_description"] = self.get_action_description(action)

        # Final end check only if not done already
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