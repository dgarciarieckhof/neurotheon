# Neurotheon

> Neurotheon is a made-up term: "neuro" for brain, "theon" for gods, so it literally means "the gods of the brain"

```bash
███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗ ████████╗██╗  ██╗███████╗ ██████╗ ███╗   ██╗
████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗╚══██╔══╝██║  ██║██╔════╝██╔═══██╗████╗  ██║
██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║   ██║   ███████║█████╗  ██║   ██║██╔██╗ ██║
██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║   ██║   ██╔══██║██╔══╝  ██║   ██║██║╚██╗██║
██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝   ██║   ██║  ██║███████╗╚██████╔╝██║ ╚████║
╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═══╝
Project Code: NRTH
```

## Problem statement
Modern friend-or-foe (FoF) logic based on static rules breaks down whenever stealth profiles, sensor clutter, or rapidly changing rules of engagement are introduced. Neurotheon explores whether:

1. Reinforcement-learning + safety gating can keep fratricide near zero.

2. Learned behaviour adapts online to adversary counter-measures.

3. The network's reasoning can be surfaced through saliency maps for operator trust.

## Key features
| Category               | Feature                                                                   | Status        |
| ---------------------- | ------------------------------------------------------------------------- | ------------- |
| **Core RL**            | Gymnasium environment, PPO training, curriculum difficulty scaling        | ✔ Implemented |
| **Visual Simulator**   | Pygame 2‑D grid, entity sprites, bullet animation, camera flash, sound FX | ✔ Implemented |
| **Safety Layer**       | Runtime ROE veto preventing friendly‑fire                                 | ✔ Implemented |
| **Explainability**     | Grad‑CAM saliency overlay on every frame                                  | ✔ Implemented |
| **Live Learning View** | Parallel viewer that reloads checkpoints and shows improving play         | ✔ Implemented |
| **MLOps**              | TensorBoard logging                                                       | ✔ Implemented |
| **Stretch (optional)** | Multi‑turret coordination via PettingZoo                                  | △ backlog     |

## Quick start
```bash
# 1. create venv & install (requires Python ≥3.10, CUDA optional)
uv sync            # uses pyproject.toml

# 2. manual smoke-test
uv run python -m sim.pygame_visual          # WASD / arrow keys to fire

# 3. start training (uses GPU if available)
uv run python -m agents.train               # checkpoints drop in ./runs/

# 4. watch learning progress in a second terminal
uv run python -m sim.live_monitor           # TAB = Grad-CAM overlay

# 5. keep track of training metrics in a third terminal
uv run tensorboard --logdir logs      
```
> **Note**: On head-less servers without audio hardware, the Pygame mixer falls back to silent mode automatically, so viewers still run over SSH-X11 or VS Code tunnels.

## Environment Details

### Turret Defense Game
The core environment is a grid-based turret defense game where:
- **Turret**: Fixed at grid center, can fire in 8 directions + idle
- **Enemies**: Red entities that advance toward turret using different pathfinding strategies
- **Allies**: Blue entities that drift randomly and must not be hit
- **Neutrals**: Yellow static entities that provide negative reward if destroyed

### Key Mechanics
- **Stealth System**: Entities have variable detection probabilities
- **Dynamic Fog-of-War**: Sensor range shrinks over time, then gradually recovers
- **Cooldown System**: Turret cannot spam shots, encouraging strategic timing
- **Pathfinding Modes**: Enemies use direct, zigzag, or spiral movement patterns
- **Reward Structure**: Starting from 0 each step, modified by actions and events

### Reward System
| Action/Event | Reward | Notes |
|-------------|--------|-------|
| Kill enemy | +3.0 + time bonus | Encourages quick elimination |
| Hit ally | -1.0 - escalating penalty | Fratricide prevention |
| Hit neutral | -0.5 | Collateral damage penalty |
| Victory (all enemies dead) | +5.0 | Mission success bonus |
| Turret hit | -0.5 per hit + time penalty | Survival incentive |
| Turret destroyed | -3.0 | Mission failure |
| Cooldown violation | -0.02 | Discourages spam |
| Idle action | -0.001 | Minimal penalty for waiting |

## Repo layout
```bash
├── assets/              # explosion.png, explosion.mp3 (optional)
├── data/
├── src/
│   ├── envs/            # turret_env.py + sensors
│   ├── agents/          # train.py, curriculum callback, stats callback
│   └── sim/             # pygame_visual.py, live_monitor.py, saliency.py
├── runs/                # checkpoints & VecNormalize stats
├── logs/                # TensorBoard event files
└── tests/               # pytest smoke tests
```

## Research Questions
1. **Adaptability**: Can the agent learn to handle evolving enemy tactics and stealth patterns?
2. **Safety**: How effectively does the safety layer prevent friendly fire while maintaining combat effectiveness?
3. **Explainability**: Do saliency maps reveal interpretable decision-making patterns that operators can trust?
4. **Generalization**: How well does the trained agent perform on unseen scenarios with different entity counts and configurations?

## Technical Architecture
- **Environment**: Custom Gymnasium environment with enhanced visual support
- **Agent**: PPO with MLP policy for grid-based decision making
- **Safety Layer**: Rule-based veto system for ROE compliance
- **Visualization**: Real-time Pygame renderer with Grad-CAM overlay
- **Training**: Curriculum learning with difficulty scaling
- **Monitoring**: Live training viewer and TensorBoard integration