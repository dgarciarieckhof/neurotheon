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
            ents.append(dict(
                pos=self._random_cell(used),
                type="enemy",
                mode=mode,
                zig_state=0,
                spiral_angle=self.rng.random() * 2 * np.pi,
                stealth=self.rng.uniform(*self.cfg["enemy_stealth"]),
            ))

        for _ in range(self.cfg["n_allies"]):
            ents.append(dict(
                pos=self._random_cell(used),
                type="ally",
                stealth=self.cfg["ally_stealth"],
            ))

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

    def _calculate_threat_level(self) -> float:
        """Calculate current threat level based on enemy positions and movement"""
        threat = 0.0
        
        for e in self.entities:
            if e["type"] == "enemy":
                distance = np.linalg.norm(e["pos"] - self.agent_pos)
                max_distance = np.sqrt(2) * self.gs  # Maximum possible distance
                
                # Higher threat for closer enemies (exponential decay)
                proximity_threat = np.exp(-distance / 5.0)
                
                # Additional threat based on movement pattern
                if e["mode"] == "direct":
                    movement_multiplier = 1.5  # Direct movement is most threatening
                elif e["mode"] == "zigzag":
                    movement_multiplier = 1.2  # Zigzag is moderately threatening
                else:  # spiral
                    movement_multiplier = 1.0  # Spiral is least predictable but slower
                
                threat += proximity_threat * movement_multiplier
        
        return threat

    def _calculate_engagement_reward(self) -> float:
        """Reward for engaging with the environment based on threat level"""
        threat = self._calculate_threat_level()
        
        # Base engagement reward scales with threat
        base_reward = 0.01 * threat
        
        # Bonus for high-threat situations
        if threat > 2.0:
            base_reward += 0.05
        elif threat > 1.0:
            base_reward += 0.02
        
        return base_reward

    def _move(self):
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

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.entities = self._spawn_entities()
        self.steps = 0
        self.sensor_range = self.sensor_range_init
        self.last_shot_step = -self.cooldown_steps
        self.turret_hits = 0
        self.ally_hit_count = 0
        self.consecutive_idle_steps = 0
        self.last_enemy_count = len([e for e in self.entities if e["type"] == "enemy"])
        return self._get_obs(), {}

    def step(self, action: int):
        info = {"hit": None, "explosion": [], "turret_destroyed": False}
        reward = 0.0
        done = False

        # Track consecutive idle actions
        if action == 0:
            self.consecutive_idle_steps += 1
        else:
            self.consecutive_idle_steps = 0

        # AGGRESSIVE IDLE PENALTY - scales with threat and consecutive idle actions
        if action == 0:
            threat_level = self._calculate_threat_level()
            
            # Base idle penalty
            idle_penalty = 0.05
            
            # Penalty scales with threat level
            threat_penalty = 0.02 * threat_level
            
            # Escalating penalty for consecutive idle actions
            consecutive_penalty = 0.01 * (self.consecutive_idle_steps ** 1.5)
            
            # Time pressure penalty (enemies get closer over time)
            time_pressure = 0.001 * (self.steps / self.max_steps)
            
            total_idle_penalty = idle_penalty + threat_penalty + consecutive_penalty + time_pressure
            reward -= total_idle_penalty
            
            # Add small engagement reward for strategic waiting in low-threat situations
            if threat_level < 0.5:
                reward += 0.005  # Very small reward for strategic patience
        
        # SHOOTING MECHANICS
        can_fire = (self.steps - self.last_shot_step) >= self.cooldown_steps
        
        if action != 0 and can_fire:
            self.last_shot_step = self.steps
            pos = self.agent_pos.copy()
            
            # Small reward for taking action (being aggressive)
            reward += 0.01
            
            # Track all hits in this shot
            hits = []
            
            # Trace the shot path - PENETRATING SHOT
            while True:
                pos = pos + self.vec[action]
                x, y = pos
                
                # Check bounds
                if not (0 <= x < self.gs and 0 <= y < self.gs):
                    break
                
                # Check for hit at this position
                hit_entities = [i for i, e in enumerate(self.entities)
                               if np.array_equal(e["pos"], pos)]
                
                if hit_entities:
                    # Process all entities at this position
                    for idx in sorted(hit_entities, reverse=True):  # Remove from end to avoid index shifts
                        e = self.entities.pop(idx)
                        hits.append(e)
                        # Add explosion for each hit entity
                        info["explosion"].append({
                            "pos": pos.copy(),
                            "type": e["type"]
                        })
            
            # Process all hits and calculate rewards
            total_shot_reward = 0.0
            enemy_kills = 0
            ally_kills = 0
            neutral_kills = 0
            
            for i, e in enumerate(hits):
                if e["type"] == "enemy":
                    enemy_kills += 1
                    
                    # Base reward for enemy kill
                    base_enemy_reward = 5.0
                    
                    # Distance bonus (closer enemies are more dangerous)
                    distance = np.linalg.norm(e["pos"] - self.agent_pos)
                    distance_bonus = max(0, 2.0 - distance * 0.2)
                    
                    # Time bonus (faster kills are better)
                    time_bonus = 0.02 * (self.max_steps - self.steps)
                    
                    # Threat level bonus
                    threat_bonus = 0.5 * self._calculate_threat_level()
                    
                    # Movement pattern bonus
                    if e["mode"] == "direct":
                        pattern_bonus = 1.0  # Most dangerous
                    elif e["mode"] == "zigzag":
                        pattern_bonus = 0.8  # Moderately dangerous
                    else:  # spiral
                        pattern_bonus = 0.6  # Least predictable
                    
                    # Multi-kill bonus (reward for efficiency)
                    multi_kill_bonus = 0.5 * (enemy_kills - 1)  # Bonus for 2nd, 3rd, etc. enemy in same shot
                    
                    total_enemy_reward = base_enemy_reward + distance_bonus + time_bonus + threat_bonus + pattern_bonus + multi_kill_bonus
                    total_shot_reward += total_enemy_reward
                    
                elif e["type"] == "ally":
                    ally_kills += 1
                    self.ally_hit_count += 1
                    
                    # Base penalty
                    base_ally_penalty = 8.0
                    
                    # Escalating penalty for multiple ally kills
                    escalation_penalty = 3.0 * (self.ally_hit_count ** 2)
                    
                    # Progress penalty (hitting allies later in the game is worse)
                    progress_penalty = 2.0 * (self.steps / self.max_steps)
                    
                    # Multi-ally kill penalty (very bad to hit multiple allies in one shot)
                    multi_ally_penalty = 5.0 * (ally_kills - 1)
                    
                    total_ally_penalty = base_ally_penalty + escalation_penalty + progress_penalty + multi_ally_penalty
                    total_shot_reward -= total_ally_penalty
                    
                else:  # neutral
                    neutral_kills += 1
                    
                    # Base penalty
                    base_neutral_penalty = 2.0
                    
                    # Small escalation for multiple neutral kills
                    neutral_count = sum(1 for e in self.entities if e["type"] == "neutral")
                    escalation = 0.5 * (self.cfg["n_neutrals"] - neutral_count)
                    
                    # Multi-neutral kill penalty
                    multi_neutral_penalty = 1.0 * (neutral_kills - 1)
                    
                    total_neutral_penalty = base_neutral_penalty + escalation + multi_neutral_penalty
                    total_shot_reward -= total_neutral_penalty
            
            # Apply the total shot reward
            reward += total_shot_reward
            
            # Set info based on what was hit
            if hits:
                if enemy_kills > 0:
                    info["hit"] = f"enemy_x{enemy_kills}" if enemy_kills > 1 else "enemy"
                elif ally_kills > 0:
                    info["hit"] = f"ally_x{ally_kills}" if ally_kills > 1 else "ally"
                    # Consider ending episode early if too many allies killed in total
                    if self.ally_hit_count >= 2:
                        reward -= 10.0  # Severe penalty
                        done = True
                        info["turret_destroyed"] = True
                else:
                    info["hit"] = f"neutral_x{neutral_kills}" if neutral_kills > 1 else "neutral"
                
                # Additional multi-target efficiency bonus
                if len(hits) > 1:
                    efficiency_bonus = 0.3 * (len(hits) - 1)
                    reward += efficiency_bonus
        
        elif action != 0 and not can_fire:
            # PENALTY FOR SHOOTING TOO FAST
            reward -= 0.1  # Increased penalty for spamming
        
        # Move entities
        self._move()

        # TURRET HIT HANDLING
        enemy_hits = [
            e for e in self.entities
            if e["type"] == "enemy" and
            np.linalg.norm(e["pos"] - self.agent_pos) <= self.kill_radius
        ]

        if enemy_hits:
            self.turret_hits += len(enemy_hits)
            
            # Add explosion for each enemy that hits the turret
            for e in enemy_hits:
                info["explosion"].append({
                    "pos": e["pos"].copy(),
                    "type": "turret_hit"
                })
            
            # Remove enemies that hit the turret
            self.entities = [
                e for e in self.entities
                if not (e["type"] == "enemy" and 
                        np.linalg.norm(e["pos"] - self.agent_pos) <= self.kill_radius)
            ]
            
            # SEVERE PENALTIES FOR TURRET HITS
            hit_penalty = 5.0 * len(enemy_hits)
            
            # Escalating penalty for multiple hits
            escalation_penalty = 2.0 * (self.turret_hits ** 2)
            
            # Progress penalty (getting hit later is worse)
            progress_penalty = 1.0 * (self.steps / self.max_steps)
            
            total_hit_penalty = hit_penalty + escalation_penalty + progress_penalty
            reward -= total_hit_penalty
            
            if self.turret_hits >= 3:
                reward -= 15.0  # Severe final penalty
                done = True
                info["turret_destroyed"] = True

        # FOG OF WAR HANDLING
        if self.fog_shrink_every and self.steps % self.fog_shrink_every == 0:
            if self.sensor_range > self.sensor_min:
                self.sensor_range -= 1
                # Small penalty for losing sensor range
                reward -= 0.05
            elif self.steps % (self.fog_shrink_every * 2) == 0:
                # Gradual recovery
                self.sensor_range = min(self.sensor_range + 1, self.sensor_range_init)
                # Small reward for regaining sensor range
                reward += 0.03

        # PROGRESS TRACKING
        current_enemy_count = len([e for e in self.entities if e["type"] == "enemy"])
        
        # Small ongoing reward for maintaining fewer enemies
        if current_enemy_count < self.last_enemy_count:
            reward += 0.5  # Progress reward
        
        self.last_enemy_count = current_enemy_count

        # TIME PRESSURE
        time_pressure_penalty = 0.005 * (self.steps / self.max_steps)
        reward -= time_pressure_penalty

        self.steps += 1

        # FINAL OUTCOME CHECKS
        if not done:
            if not any(e["type"] == "enemy" for e in self.entities):
                # VICTORY REWARD
                victory_reward = 20.0
                
                # Time bonus for fast completion
                time_bonus = 5.0 * (self.max_steps - self.steps) / self.max_steps
                
                # Ally preservation bonus
                remaining_allies = len([e for e in self.entities if e["type"] == "ally"])
                ally_bonus = 2.0 * remaining_allies
                
                total_victory_reward = victory_reward + time_bonus + ally_bonus
                reward += total_victory_reward
                done = True
                
            elif self.steps >= self.max_steps:
                # TIMEOUT PENALTY
                remaining_enemies = len([e for e in self.entities if e["type"] == "enemy"])
                timeout_penalty = 3.0 * remaining_enemies
                reward -= timeout_penalty
                done = True

        return self._get_obs(), reward, done, False, info

    def render(self):
        pass