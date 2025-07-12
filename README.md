# Neurotheon

> Neurotheon is a made-up term: “neuro” for brain, “theon” for gods, so it literally means “the gods of the brain”

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
Modern friend-or-foe (FoF) logic based on static rules breaks down
whenever stealth profiles, sensor clutter, or rapidly changing rules of
engagement are introduced. Neurotheon explores whether:

1. Reinforcement-learning + safety gating can keep fratricide near
zero,

2. Learned behaviour adapts online to adversary counter-measures,

3. The network’s reasoning can be surfaced through saliency maps for operator trust.

## Key features
| Category               | Feature                                                                   | Status        |
| ---------------------- | ------------------------------------------------------------------------- | ------------- |
| **Core RL**            | Gymnasium environment, PPO training, curriculum difficulty scaling        | ✔ Implemented |
| **Visual Simulator**   | Pygame 2‑D grid, entity sprites, bullet animation, camera flash, sound FX | ✔ Implemented |
| **Safety Layer**       | Runtime ROE veto preventing friendly‑fire                                 | ✔ Implemented |
| **Explainability**     | Grad‑CAM saliency overlay on every frame                                  | ✔ Implemented |
| **Live Learning View** | Parallel viewer that reloads checkpoints and shows improving play         | ✔ Implemented |
| **MLOps**              | TensorBoard logging       | ✔ Implemented |
| **Stretch (optional)** | Multi‑turret coordination via PettingZoo                                  | △ backlog     |

## Quick start
```bash
# 1. create venv & install (requires Python ≥3.10, CUDA optional)
uv sync            # uses pyproject.toml

# 2. manual smoke-test
python -m sim.pygame_visual          # WASD / arrow keys to fire

# 3. start training (uses GPU if available)
python -m agents.train               # checkpoints drop in ./runs/

# 4. watch learning progress in a second terminal
python -m sim.live_monitor           # TAB = Grad-CAM overlay

# 5. keep track of training metrics in a third terminal
tensorboard --logdir logs --port 6006           
```
> **Note**: On head-less servers without audio hardware, the Pygame mixer falls back to silent mode automatically, so viewers still run over SSH-X11 or VS Code tunnels.

## Repo layout
```bash
├── assets/              # explosion.png, explosion.mp3 (optional)
├── data/
├── src/
│   ├── envs/            # turret_env.py  + sensors
│   ├── agents/          # train.py, curriculum callback, stats callback
│   └── sim/             # pygame_visual.py, live_monitor.py, saliency.py
├── runs/                # checkpoints & VecNormalize stats
├── logs/                # TensorBoard event files
└── tests/               # pytest smoke tests
```