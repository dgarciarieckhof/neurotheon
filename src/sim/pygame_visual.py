# sim/pygame_visual.py  – polished night-theme viewer
import sys, pygame, pathlib, math
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

# Color palette
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
    shot_trail=(255, 255, 255, 100),  # Semi-transparent white for shot trails
)

KEY2ACT = {
    pygame.K_UP: 1, pygame.K_DOWN: 2, pygame.K_LEFT: 3, pygame.K_RIGHT: 4,
    pygame.K_q: 5, pygame.K_e: 6, pygame.K_z: 7, pygame.K_c: 8,
    pygame.K_SPACE: 0
}

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

# ─────── main drawing function ──────────────────────────────

def draw(env, surf, font, fps, step, score, kills, time_left,
         explosions, shake_off, mission_banner=None, shot_trail=None):
    """Main drawing function with improved organization"""
    
    gs = env.gs
    surf.fill(COL["bg"])
    
    # Calculate shake offset
    off_x, off_y = calculate_shake_offset(shake_off)
    
    # Draw grid background
    draw_grid(surf, gs, off_x, off_y)
    
    # Draw shot trail if present
    if shot_trail:
        draw_shot_trail(surf, shot_trail, off_x, off_y)
    
    # Draw entities
    draw_entities(surf, env, off_x, off_y)
    
    # Draw turret
    draw_turret(surf, env, off_x, off_y)
    
    # Draw explosions
    draw_explosions(surf, explosions, EXP_FRAMES, off_x, off_y)
    
    # Draw HUD
    draw_hud(surf, font, fps, step, env.max_steps, score, kills, time_left, env.turret_hits)
    
    # Draw mission banner if present
    if mission_banner:
        draw_mission_banner(surf, font, mission_banner)
    
    pygame.display.flip()
    
    return max(0, shake_off - 1)

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

def draw_shot_trail(surf, trail_data, off_x, off_y):
    """Draw shot trail visualization"""
    if not trail_data or len(trail_data) < 2:
        return
    
    # Create trail surface with alpha
    trail_surf = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    
    # Draw trail as connected lines with decreasing alpha
    for i in range(len(trail_data) - 1):
        start_pos = trail_data[i]
        end_pos = trail_data[i + 1]
        
        # Convert grid coordinates to pixel coordinates
        start_pixel = (start_pos[1] * CELL + CELL // 2 + off_x, 
                      start_pos[0] * CELL + CELL // 2 + off_y)
        end_pixel = (end_pos[1] * CELL + CELL // 2 + off_x, 
                    end_pos[0] * CELL + CELL // 2 + off_y)
        
        # Calculate alpha based on position in trail (newer = more opaque)
        alpha = int(255 * (i + 1) / len(trail_data))
        color = (*COL["shot_trail"][:3], alpha)
        
        # Draw line segment
        pygame.draw.line(trail_surf, color, start_pixel, end_pixel, 3)
    
    surf.blit(trail_surf, (0, 0))

def draw_entities(surf, env, off_x, off_y):
    """Draw all entities (enemies, allies, neutrals)"""
    entity_colors = {"enemy": "enm", "ally": "ally", "neutral": "neu"}
    
    for e in env.entities:
        x, y = e["pos"]
        col = COL[entity_colors[e["type"]]]
        entity_rect = shadow_rect(x, y).move(off_x, off_y)
        pygame.draw.rect(surf, col, entity_rect)
        
        # Add border for better visibility
        pygame.draw.rect(surf, (255, 255, 255), entity_rect, 1)
        
        # Add visual indicators for enemy movement patterns
        if e["type"] == "enemy":
            draw_enemy_indicator(surf, e, entity_rect)

def draw_enemy_indicator(surf, enemy, rect):
    """Draw small indicators for enemy movement patterns"""
    center_x = rect.centerx
    center_y = rect.centery
    
    if enemy["mode"] == "direct":
        # Draw arrow pointing toward turret
        pygame.draw.polygon(surf, (255, 255, 255), [
            (center_x, center_y - 4),
            (center_x - 3, center_y + 2),
            (center_x + 3, center_y + 2)
        ])
    elif enemy["mode"] == "zigzag":
        # Draw zigzag pattern
        points = [
            (center_x - 4, center_y - 2),
            (center_x, center_y + 2),
            (center_x + 4, center_y - 2)
        ]
        pygame.draw.lines(surf, (255, 255, 255), False, points, 2)
    elif enemy["mode"] == "spiral":
        # Draw spiral indicator
        pygame.draw.circle(surf, (255, 255, 255), (center_x, center_y), 3, 1)
        pygame.draw.circle(surf, (255, 255, 255), (center_x, center_y), 1)

def draw_turret(surf, env, off_x, off_y):
    """Draw the player turret"""
    tx, ty = env.agent_pos
    turret_rect = shadow_rect(tx, ty).move(off_x, off_y)
    pygame.draw.rect(surf, COL["tur"], turret_rect)
    
    # Add turret border
    pygame.draw.rect(surf, (255, 255, 255), turret_rect, 2)
    
    # Draw sensor range indicator if desired
    if hasattr(env, 'sensor_range') and env.sensor_range > 0:
        center_x = turret_rect.centerx
        center_y = turret_rect.centery
        sensor_radius = env.sensor_range * CELL
        
        # Draw sensor range circle (very faint)
        sensor_surf = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        pygame.draw.circle(sensor_surf, (100, 100, 255, 30), 
                          (center_x, center_y), sensor_radius, 1)
        surf.blit(sensor_surf, (0, 0))

def draw_explosions(surf, explosions, exp_frames, off_x, off_y):
    """Draw explosion animations - handles both single positions and lists"""
    if not explosions:
        return
    
    for ex in explosions[:]:  # Use slice to avoid modification during iteration
        if ex["frame"] >= len(exp_frames):
            explosions.remove(ex)
            continue
        
        img = exp_frames[ex["frame"]]
        
        # Handle both single position and list of positions
        positions = ex["pos"]
        if not isinstance(positions, list):
            positions = [positions]
        
        # Draw explosion at each position
        for pos in positions:
            if isinstance(pos, (list, tuple)) and len(pos) == 2:
                px, py = pos
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
    
    # Draw additional stats
    if hasattr(font, 'render'):
        small_font = pygame.font.SysFont('Arial', 18)
        
        # Sensor range info
        if hasattr(surf, '_env_ref'):
            env = surf._env_ref
            if hasattr(env, 'sensor_range'):
                sensor_text = f"Sensor Range: {env.sensor_range}"
                draw_text_with_background(surf, small_font, sensor_text, (6, 55))
        
        # Cooldown indicator
        cooldown_text = "READY" if time_left > 0 else "COOLDOWN"
        cooldown_color = (0, 255, 0) if time_left > 0 else (255, 0, 0)
        draw_text_with_background(surf, small_font, cooldown_text, (6, 75), cooldown_color)

def draw_mission_banner(surf, font, banner_text):
    """Draw mission banner at the top of the screen"""
    banner_surf = pygame.Surface((surf.get_width(), 40), pygame.SRCALPHA)
    banner_surf.fill((0, 0, 0, 180))
    
    text_surf = font.render(banner_text, True, (255, 255, 255))
    text_rect = text_surf.get_rect(center=(surf.get_width() // 2, 20))
    
    banner_surf.blit(text_surf, text_rect)
    surf.blit(banner_surf, (0, 0))

# ─────── main viewer ──────────────────────────

def main():
    """Main game loop"""
    pygame.init()
    pygame.mixer.init()
    
    # Initialize game
    env = TurretEnv()
    surf = pygame.display.set_mode((env.gs*CELL, env.gs*CELL))
    pygame.display.set_caption("Turret Defense - Night Theme")
    font = pygame.font.SysFont('Arial', 24)  # Use Arial for better readability
    clock = pygame.time.Clock()
    
    # Game state
    obs, _ = env.reset()
    env.observation = obs
    last_act = 0
    step = score = kills = 0
    explosions = []
    shake_off = 0
    shot_trail = []
    
    # Store env reference for HUD
    surf._env_ref = env
    
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
                    shot_trail.clear()
                elif e.key in (pygame.K_ESCAPE, pygame.K_x):
                    running = False
        
        # Clear shot trail from previous frame
        shot_trail.clear()
        
        # Update game state
        obs, r, done, _, info = env.step(last_act)
        env.observation = obs
        step += 1
        score += r
        
        # Handle game events
        hit_info = info.get("hit", "")
        if "enemy" in hit_info:
            # Extract number of kills from hit_info
            if "x" in hit_info:
                kills += int(hit_info.split("x")[1])
            else:
                kills += 1
        
        # Handle explosions (now can be a list of positions)
        explosion_positions = info.get("explosion")
        if explosion_positions:
            if isinstance(explosion_positions, list):
                # Multiple explosions
                for pos in explosion_positions:
                    explosions.append(dict(pos=pos, frame=0))
            else:
                # Single explosion (backward compatibility)
                explosions.append(dict(pos=explosion_positions, frame=0))
            
            shake_off = SHAKE_MS // (1000 // FPS)
            if SND_EXPLO:
                SND_EXPLO.play()
        
        # Create shot trail visualization if action was taken
        if last_act != 0:
            # Simple trail from turret position outward
            shot_trail = create_shot_trail(env, last_act)
        
        # Draw everything
        shake_off = draw(
            env, surf, font, clock.get_fps(),
            step, score, kills,
            env.max_steps - step,
            explosions, shake_off,
            shot_trail=shot_trail
        )
        
        clock.tick(FPS)
        
        # Reset on game over
        if done:
            obs, _ = env.reset()
            env.observation = obs
            last_act = step = score = kills = 0
            explosions.clear()
            shake_off = 0
            shot_trail.clear()
    
    pygame.quit()
    sys.exit()

def create_shot_trail(env, action):
    """Create a shot trail for visualization"""
    if action == 0:
        return []
    
    trail = []
    pos = env.agent_pos.copy()
    trail.append(pos.copy())
    
    # Trace the shot path
    vec = env.vec[action]
    for _ in range(max(env.gs, 50)):  # Limit iterations
        pos = pos + vec
        if not (0 <= pos[0] < env.gs and 0 <= pos[1] < env.gs):
            break
        trail.append(pos.copy())
        
        # Stop if we hit something (optional - depends on if you want to show full trail)
        # This is simplified - in reality you'd check entity positions
    
    return trail

if __name__ == "__main__":
    main()