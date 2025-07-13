# sim/pygame_visual.py  – polished night-theme viewer with enhanced visuals
import sys, pygame, pathlib, math
import numpy as np
from envs.turret_env import TurretEnv

# ─────── assets & constants ────────────────────────────────
CELL, MARGIN, FPS = 40, 2, 20
SHAKE_MS, SHAKE_PIX = 50, 4

ASSET_DIR   = pathlib.Path(__file__).resolve().parents[2] / "assets"

# Load assets with error handling
def load_assets():
    """Load game assets with proper error handling"""
    exp_sheet = None
    exp_frames = []
    snd_explo = None
    
    try:
        exp_sheet = pygame.image.load(ASSET_DIR / "explosion.png")
        exp_frames = [exp_sheet.subsurface(pygame.Rect(col*32, row*32, 32, 32))
                     for row in range(4) for col in range(4)]
        exp_frames = [pygame.transform.smoothscale(f, (CELL, CELL))
                      for f in exp_frames]
    except (pygame.error, FileNotFoundError) as e:
        print(f"Warning: Could not load explosion sprite: {e}")
        # Create simple colored rectangles as fallback
        exp_frames = [pygame.Surface((CELL, CELL)) for _ in range(16)]
        for i, frame in enumerate(exp_frames):
            intensity = 255 - (i * 15)  # Fade out effect
            frame.fill((intensity, intensity // 2, 0))

    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        snd_explo = pygame.mixer.Sound(ASSET_DIR / "explosion.mp3")
    except (pygame.error, FileNotFoundError) as e:
        print(f"Warning: Could not load explosion sound: {e}")
        snd_explo = None

    return exp_frames, snd_explo

# Load assets at module level for backward compatibility
EXP_FRAMES, SND_EXPLO = load_assets()

# Color palette - original colors plus new additions
COL = dict(
    bg   =(0, 0, 0),
    grid =(40, 40, 40),      # Darker grid for better visibility
    und  =(25, 25, 25),
    sh   =(0, 0, 0, 80),
    tur  =(50, 120, 255),
    enm  =(255, 60, 60),
    ally =(0, 255, 150),
    neu  =(255, 255, 60),
    text =(255, 255, 255),
    text_bg=(0, 0, 0, 128),  # Semi-transparent background for text
    # New visual effect colors
    laser=(255, 100, 255),      # Bright magenta for laser shots
    muzzle=(255, 255, 0),       # Yellow for muzzle flash
    sensor=(0, 255, 255, 30),   # Cyan for sensor range
    warning=(255, 100, 100),    # Red warning color
    trail=(255, 255, 255, 120), # White trail for projectiles
    glow=(255, 255, 255, 50),   # Soft glow effect
)

KEY2ACT = {
    pygame.K_UP: 1, pygame.K_DOWN: 2, pygame.K_LEFT: 3, pygame.K_RIGHT: 4,
    pygame.K_q: 5, pygame.K_e: 6, pygame.K_z: 7, pygame.K_c: 8,
    pygame.K_SPACE: 0
}

# ─────── Visual Effects System ─────────────────────────────

class VisualEffects:
    def __init__(self):
        self.laser_shots = []      # Active laser shots
        self.muzzle_flashes = []   # Muzzle flash effects
        self.particle_effects = [] # Particle systems
        self.warning_pulses = []   # Warning indicators
        self.entity_trails = {}    # Movement trails for entities
        
    def update(self):
        # Update all visual effects
        self.laser_shots = [shot for shot in self.laser_shots if shot['life'] > 0]
        self.muzzle_flashes = [flash for flash in self.muzzle_flashes if flash['life'] > 0]
        self.particle_effects = [p for p in self.particle_effects if p['life'] > 0]
        self.warning_pulses = [w for w in self.warning_pulses if w['life'] > 0]
        
        # Update lifetimes
        for effect_list in [self.laser_shots, self.muzzle_flashes, self.particle_effects, self.warning_pulses]:
            for effect in effect_list:
                effect['life'] -= 1

# ─────── helper functions ───────────────────────────────

def shadow_rect(x, y):
    """Create a rectangle with margin for shadows"""
    return pygame.Rect(y*CELL + MARGIN, x*CELL + MARGIN,
                       CELL - 2*MARGIN, CELL - 2*MARGIN)

def calculate_shake_offset(shake_off):
    """Calculate screen shake offset with smooth decay"""
    if shake_off <= 0:
        return 0, 0
    
    # Use sine waves for smooth shake
    off_x = int(math.sin(shake_off * 31) * SHAKE_PIX * (shake_off / (SHAKE_MS // (1000 // FPS))))
    off_y = int(math.cos(shake_off * 17) * SHAKE_PIX * (shake_off / (SHAKE_MS // (1000 // FPS))))
    return off_x, off_y

def draw_text_with_background(surf, font, text, pos, text_color=COL["text"], bg_color=COL["text_bg"]):
    """Draw text with semi-transparent background for better readability"""
    text_surf = font.render(text, True, text_color)
    text_rect = text_surf.get_rect(topleft=pos)
    
    # Create background surface
    bg_surf = pygame.Surface((text_rect.width + 8, text_rect.height + 4), pygame.SRCALPHA)
    bg_surf.fill(bg_color)
    
    surf.blit(bg_surf, (text_rect.x - 4, text_rect.y - 2))
    surf.blit(text_surf, pos)

# ─────── Enhanced Drawing Functions ─────────────────────────────

def draw_sensor_range(surf, env, off_x, off_y, alpha=30):
    """Draw turret sensor range as a semi-transparent circle"""
    tx, ty = env.agent_pos
    center = (ty * CELL + CELL // 2 + off_x, tx * CELL + CELL // 2 + off_y)
    radius = env.sensor_range * CELL
    
    # Create a surface for the sensor range with alpha
    sensor_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    pygame.draw.circle(sensor_surf, (0, 255, 255, alpha), (radius, radius), radius)
    
    # Blit centered on turret
    surf.blit(sensor_surf, (center[0] - radius, center[1] - radius))

def draw_laser_shots(surf, laser_shots, off_x, off_y):
    """Draw laser shot effects"""
    for shot in laser_shots:
        start_pos = (shot['start'][1] * CELL + CELL // 2 + off_x, 
                    shot['start'][0] * CELL + CELL // 2 + off_y)
        end_pos = (shot['end'][1] * CELL + CELL // 2 + off_x, 
                  shot['end'][0] * CELL + CELL // 2 + off_y)
        
        # Draw laser beam with thickness based on remaining life
        thickness = max(1, shot['life'] // 2)
        pygame.draw.line(surf, COL["laser"], start_pos, end_pos, thickness)
        
        # Add glow effect
        if thickness > 1:
            pygame.draw.line(surf, (255, 255, 255, 100), start_pos, end_pos, thickness + 2)

def draw_muzzle_flashes(surf, muzzle_flashes, off_x, off_y):
    """Draw muzzle flash effects"""
    for flash in muzzle_flashes:
        center = (flash['pos'][1] * CELL + CELL // 2 + off_x,
                 flash['pos'][0] * CELL + CELL // 2 + off_y)
        
        # Flash size decreases with life
        size = flash['life'] * 3
        if size > 0:
            # Draw bright flash
            pygame.draw.circle(surf, COL["muzzle"], center, size)
            # Add white core
            pygame.draw.circle(surf, (255, 255, 255), center, size // 2)

def draw_entity_trails(surf, entity_trails, off_x, off_y):
    """Draw movement trails for entities"""
    for entity_id, trail in entity_trails.items():
        if len(trail) > 1:
            # Draw trail as connected lines with fading alpha
            for i in range(len(trail) - 1):
                alpha = int(255 * (i / len(trail)))
                start_pos = (trail[i][1] * CELL + CELL // 2 + off_x,
                           trail[i][0] * CELL + CELL // 2 + off_y)
                end_pos = (trail[i+1][1] * CELL + CELL // 2 + off_x,
                          trail[i+1][0] * CELL + CELL // 2 + off_y)
                
                # Create surface for alpha blending
                trail_surf = pygame.Surface((abs(end_pos[0] - start_pos[0]) + 10,
                                          abs(end_pos[1] - start_pos[1]) + 10), pygame.SRCALPHA)
                rel_start = (5, 5)
                rel_end = (end_pos[0] - start_pos[0] + 5, end_pos[1] - start_pos[1] + 5)
                pygame.draw.line(trail_surf, (255, 255, 255, alpha), rel_start, rel_end, 2)
                surf.blit(trail_surf, (min(start_pos[0], end_pos[0]) - 5,
                                     min(start_pos[1], end_pos[1]) - 5))

def draw_warning_indicators(surf, warning_pulses, off_x, off_y):
    """Draw warning indicators for dangerous enemies"""
    for warning in warning_pulses:
        center = (warning['pos'][1] * CELL + CELL // 2 + off_x,
                 warning['pos'][0] * CELL + CELL // 2 + off_y)
        
        # Pulsing warning circle
        pulse_size = 20 + int(10 * math.sin(warning['life'] * 0.5))
        alpha = warning['life'] * 3
        
        # Create warning surface
        warning_surf = pygame.Surface((pulse_size * 2, pulse_size * 2), pygame.SRCALPHA)
        pygame.draw.circle(warning_surf, (255, 100, 100, alpha), 
                         (pulse_size, pulse_size), pulse_size, 3)
        surf.blit(warning_surf, (center[0] - pulse_size, center[1] - pulse_size))

def draw_enhanced_entities(surf, env, off_x, off_y, visual_effects):
    """Enhanced entity drawing with glows and animations"""
    for i, e in enumerate(env.entities):
        x, y = e["pos"]
        
        # Update entity trails
        entity_id = id(e)
        if entity_id not in visual_effects.entity_trails:
            visual_effects.entity_trails[entity_id] = []
        
        trail = visual_effects.entity_trails[entity_id]
        trail.append((x, y))
        if len(trail) > 8:  # Keep last 8 positions
            trail.pop(0)
        
        # Draw entity with enhanced visuals
        entity_rect = shadow_rect(x, y).move(off_x, off_y)
        
        # Add glow effect for enemies
        if e["type"] == "enemy":
            glow_surf = pygame.Surface((CELL + 10, CELL + 10), pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (255, 60, 60, 30), 
                           pygame.Rect(5, 5, CELL, CELL))
            surf.blit(glow_surf, (entity_rect.x - 5, entity_rect.y - 5))
            
            # Add warning pulse for nearby enemies
            dist_to_turret = math.sqrt((x - env.agent_pos[0])**2 + (y - env.agent_pos[1])**2)
            if dist_to_turret < 5:  # Close to turret
                visual_effects.warning_pulses.append({
                    'pos': (x, y),
                    'life': 30
                })
        
        # Draw the entity
        col = COL["enm"] if e["type"] == "enemy" else COL["ally"] if e["type"] == "ally" else COL["neu"]
        pygame.draw.rect(surf, col, entity_rect)
        
        # Add animated border
        border_color = (255, 255, 255)
        if e["type"] == "enemy":
            # Pulsing red border for enemies
            pulse = int(128 + 127 * math.sin(pygame.time.get_ticks() * 0.01))
            border_color = (pulse, pulse // 2, pulse // 2)
        
        pygame.draw.rect(surf, border_color, entity_rect, 2)

def draw_enhanced_turret(surf, env, off_x, off_y):
    """Enhanced turret drawing with rotation and effects"""
    tx, ty = env.agent_pos
    turret_rect = shadow_rect(tx, ty).move(off_x, off_y)
    
    # Add turret glow
    glow_surf = pygame.Surface((CELL + 20, CELL + 20), pygame.SRCALPHA)
    pygame.draw.rect(glow_surf, (50, 120, 255, 40), 
                   pygame.Rect(10, 10, CELL, CELL))
    surf.blit(glow_surf, (turret_rect.x - 10, turret_rect.y - 10))
    
    # Draw turret base
    pygame.draw.rect(surf, COL["tur"], turret_rect)
    
    # Draw turret barrel (pointing towards nearest enemy)
    center = turret_rect.center
    nearest_enemy = None
    min_dist = float('inf')
    
    for e in env.entities:
        if e["type"] == "enemy":
            dist = math.sqrt((e["pos"][0] - tx)**2 + (e["pos"][1] - ty)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_enemy = e
    
    if nearest_enemy:
        # Calculate angle to nearest enemy
        dx = nearest_enemy["pos"][1] - ty
        dy = nearest_enemy["pos"][0] - tx
        angle = math.atan2(dy, dx)
        
        # Draw barrel
        barrel_length = CELL // 2
        end_x = center[0] + barrel_length * math.cos(angle)
        end_y = center[1] + barrel_length * math.sin(angle)
        
        pygame.draw.line(surf, (255, 255, 255), center, (end_x, end_y), 4)
        pygame.draw.line(surf, COL["tur"], center, (end_x, end_y), 2)
    
    # Enhanced turret border
    pygame.draw.rect(surf, (255, 255, 255), turret_rect, 3)

# ─────── Original Drawing Functions (Enhanced) ─────────────────────────────

def draw_grid(surf, gs, off_x, off_y):
    """Draw the game grid"""
    # Draw cell backgrounds
    grid_surf = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    for x in range(gs):
        for y in range(gs):
            pygame.draw.rect(grid_surf, COL["und"], shadow_rect(x, y))
            pygame.draw.rect(grid_surf, COL["sh"], shadow_rect(x, y))
    surf.blit(grid_surf, (off_x, off_y))
    
    # Draw grid lines
    for i in range(gs + 1):
        x = i * CELL + off_x
        y = i * CELL + off_y
        pygame.draw.line(surf, COL["grid"], (x, 0), (x, gs*CELL), 1)
        pygame.draw.line(surf, COL["grid"], (0, y), (gs*CELL, y), 1)

def draw_explosions(surf, explosions, exp_frames, off_x, off_y):
    """Draw explosion animations"""
    for ex in explosions[:]:
        if ex["frame"] >= len(exp_frames):
            explosions.remove(ex)
            continue
        
        img = exp_frames[ex["frame"]]
        px, py = ex["pos"]
        surf.blit(img, (py*CELL + off_x, px*CELL + off_y))
        ex["frame"] += 1

def draw_hud(surf, font, fps, step, max_steps, score, kills, time_left, turret_hits):
    """Draw the heads-up display"""
    max_hits = 3
    hits_left = max_hits - turret_hits
    
    # Main HUD line
    hud_text = (f"FPS {fps:>4.1f}   step {step}/{max_steps}   "
                f"reward {score:6.2f}   kills {kills}   T-{time_left}   "
                f"health {hits_left}/{max_hits}")
    
    draw_text_with_background(surf, font, hud_text, (6, 4))
    
    # Health bar visualization
    health_bar_width = 100
    health_bar_height = 8
    health_ratio = hits_left / max_hits
    
    # Health bar background
    health_bg_rect = pygame.Rect(6, 35, health_bar_width, health_bar_height)
    pygame.draw.rect(surf, (60, 60, 60), health_bg_rect)
    
    # Health bar fill
    health_fill_width = int(health_bar_width * health_ratio)
    health_fill_rect = pygame.Rect(6, 35, health_fill_width, health_bar_height)
    
    # Color based on health level
    if health_ratio > 0.66:
        health_color = (0, 255, 0)
    elif health_ratio > 0.33:
        health_color = (255, 255, 0)
    else:
        health_color = (255, 0, 0)
    
    pygame.draw.rect(surf, health_color, health_fill_rect)
    pygame.draw.rect(surf, (255, 255, 255), health_bg_rect, 1)

# ─────── Main Enhanced Drawing Function ─────────────────────────────

def draw_enhanced(env, surf, font, fps, step, score, kills, time_left,
                 explosions, shake_off, visual_effects, mission_banner=None):
    """Enhanced drawing function with all visual effects"""
    
    gs = env.gs
    surf.fill(COL["bg"])
    
    # Calculate shake offset
    off_x, off_y = calculate_shake_offset(shake_off)
    
    # Draw grid background
    draw_grid(surf, gs, off_x, off_y)
    
    # Draw sensor range
    draw_sensor_range(surf, env, off_x, off_y)
    
    # Draw entity trails
    draw_entity_trails(surf, visual_effects.entity_trails, off_x, off_y)
    
    # Draw enhanced entities
    draw_enhanced_entities(surf, env, off_x, off_y, visual_effects)
    
    # Draw enhanced turret
    draw_enhanced_turret(surf, env, off_x, off_y)
    
    # Draw visual effects
    draw_laser_shots(surf, visual_effects.laser_shots, off_x, off_y)
    draw_muzzle_flashes(surf, visual_effects.muzzle_flashes, off_x, off_y)
    draw_warning_indicators(surf, visual_effects.warning_pulses, off_x, off_y)
    
    # Draw explosions
    draw_explosions(surf, explosions, EXP_FRAMES, off_x, off_y)
    
    # Draw HUD
    draw_hud(surf, font, fps, step, env.max_steps, score, kills, time_left, env.turret_hits)
    
    # Update visual effects
    visual_effects.update()
    
    pygame.display.flip()
    
    return max(0, shake_off - 1)

# ─────── Visual Effects Handler ─────────────────────────────

def handle_shooting_effects(env, info, last_act, visual_effects):
    """Add visual effects when shooting"""
    if last_act != 0:  # If shooting
        # Add muzzle flash
        visual_effects.muzzle_flashes.append({
            'pos': env.agent_pos.copy(),
            'life': 8
        })
        
        # Add laser shot effect
        direction = env.vec[last_act]
        start_pos = env.agent_pos.copy()
        end_pos = start_pos.copy()
        
        # Calculate laser end position
        while True:
            end_pos = end_pos + direction
            if not (0 <= end_pos[0] < env.gs and 0 <= end_pos[1] < env.gs):
                break
            # Check if hit something
            if any(np.array_equal(e["pos"], end_pos) for e in env.entities):
                break
        
        visual_effects.laser_shots.append({
            'start': start_pos,
            'end': end_pos,
            'life': 10
        })

# ─────── Original draw function (for backward compatibility) ─────────────────

def draw(env, surf, font, fps, step, score, kills, time_left,
         explosions, shake_off, mission_banner=None):
    """Main drawing function with improved organization"""
    
    gs = env.gs
    surf.fill(COL["bg"])
    
    # Calculate shake offset
    off_x, off_y = calculate_shake_offset(shake_off)
    
    # Draw grid background
    draw_grid(surf, gs, off_x, off_y)
    
    # Draw entities (original version)
    draw_entities(surf, env, off_x, off_y)
    
    # Draw turret (original version)
    draw_turret(surf, env, off_x, off_y)
    
    # Draw explosions
    draw_explosions(surf, explosions, EXP_FRAMES, off_x, off_y)
    
    # Draw HUD
    draw_hud(surf, font, fps, step, env.max_steps, score, kills, time_left, env.turret_hits)
    
    pygame.display.flip()
    
    return max(0, shake_off - 1)

def draw_entities(surf, env, off_x, off_y):
    """Draw all entities (enemies, allies, neutrals) - original version"""
    entity_colors = {"enemy": "enm", "ally": "ally", "neutral": "neu"}
    
    for e in env.entities:
        x, y = e["pos"]
        col = COL[entity_colors[e["type"]]]
        entity_rect = shadow_rect(x, y).move(off_x, off_y)
        pygame.draw.rect(surf, col, entity_rect)
        
        # Add border for better visibility
        pygame.draw.rect(surf, (255, 255, 255), entity_rect, 1)

def draw_turret(surf, env, off_x, off_y):
    """Draw the player turret - original version"""
    tx, ty = env.agent_pos
    turret_rect = shadow_rect(tx, ty).move(off_x, off_y)
    pygame.draw.rect(surf, COL["tur"], turret_rect)
    
    # Add turret border
    pygame.draw.rect(surf, (255, 255, 255), turret_rect, 2)

# ─────── Enhanced Main Game Loop ─────────────────────────────

def main():
    """Main game loop with enhanced visuals"""
    pygame.init()
    pygame.mixer.init()
    
    # Initialize game
    env = TurretEnv()
    surf = pygame.display.set_mode((env.gs*CELL, env.gs*CELL))
    pygame.display.set_caption("Turret Defense - Enhanced Night Theme")
    font = pygame.font.SysFont('Arial', 24)  # Use Arial for better readability
    clock = pygame.time.Clock()
    
    # Initialize visual effects system
    visual_effects = VisualEffects()
    
    # Game state
    obs, _ = env.reset()
    env.observation = obs
    last_act = 0
    step = score = kills = 0
    explosions = []
    shake_off = 0
    
    # Game loop
    running = True
    while running:
        # Handle events
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in KEY2ACT:
                    last_act = KEY2ACT[e.key]
                elif e.key == pygame.K_r:
                    # Reset game
                    obs, _ = env.reset()
                    env.observation = obs
                    last_act = step = score = kills = 0
                    explosions.clear()
                    shake_off = 0
                    visual_effects = VisualEffects()  # Reset visual effects
                elif e.key in (pygame.K_ESCAPE, pygame.K_x):
                    running = False
        
        # Update game state
        obs, r, done, _, info = env.step(last_act)
        env.observation = obs
        step += 1
        score += r
        
        # Handle shooting effects
        handle_shooting_effects(env, info, last_act, visual_effects)
        
        # Handle game events
        if info.get("hit") == "enemy":
            kills += 1
        
        if info.get("explosion"):
            explosions.append(dict(pos=info["explosion"], frame=0))
            shake_off = SHAKE_MS // (1000 // FPS)
            if SND_EXPLO:
                SND_EXPLO.play()
        
        # Draw everything with enhancements
        shake_off = draw_enhanced(
            env, surf, font, clock.get_fps(),
            step, score, kills,
            env.max_steps - step,
            explosions, shake_off, visual_effects
        )
        
        clock.tick(FPS)
        
        # Reset on game over
        if done:
            obs, _ = env.reset()
            env.observation = obs
            last_act = step = score = kills = 0
            explosions.clear()
            shake_off = 0
            visual_effects = VisualEffects()  # Reset visual effects
    
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()