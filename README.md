# Neurotheon

```bash
| \ | | _____ _____ _ | | _ __ _ __()_ __ __ _
| | |/ _ \ / / _ \ '| _| ' | '| | '_ \ / ` |
| |\ | __/> < __/ | | || |) | | | | | | | (| |_
|| _|___//__|| _| ./|| ||| ||_, ()
|| |__/
NEUROTHEON · Project Code: NRTH
```

## Problem statement
Conventional rule‑based friend‑or‑foe logic struggles when stealth profiles, clutter, or ROE constraints shift. Lockheed Martin seeks proof that reinforcement‑learning + safety gating can (1) maintain near‑zero fratricide, (2) adapt to adversary counter‑measures, and (3) expose reasoning to operators/inspectors.

## Key features
| Category               | Feature                                                                   | Status        |
| ---------------------- | ------------------------------------------------------------------------- | ------------- |
| **Core RL**            | Gymnasium environment, PPO training, curriculum difficulty scaling        | ✔ Implemented |
| **Visual Simulator**   | Pygame 2‑D grid, entity sprites, bullet animation, camera flash, sound FX | ✔ Implemented |
| **Safety Layer**       | Runtime ROE veto preventing friendly‑fire                                 | ✔ Implemented |
| **Explainability**     | Grad‑CAM saliency overlay on every frame                                  | ✔ Implemented |
| **Live Learning View** | Parallel viewer that reloads checkpoints and shows improving play         | ✔ Implemented |
| **MLOps**              | Hydra configs, WandB/TensorBoard logging, Docker, GitHub Actions CI       | ✔ Implemented |
| **Stretch (optional)** | Multi‑turret coordination via PettingZoo                                  | △ backlog     |
