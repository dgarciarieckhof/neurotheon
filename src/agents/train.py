"""
Train a PPO agent on TurretEnv with curriculum and GPU auto-selection.
Checkpoints every 10k steps in `runs/`.
"""

from envs.turret_env import TurretEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from agents.curriculum import CurriculumCallback

TOTAL_STEPS = 150_000
CKPT_FREQ   = 10_000

def make_env(**kw):
    return TurretEnv(**kw)

# easy start
env = make_env(n_enemies=4, enemy_move_every=7)

# callbacks
ckpt_cb = CheckpointCallback(
    save_freq=CKPT_FREQ, save_path="runs", name_prefix="ppo_turret_step"
)
curric_cb = CurriculumCallback(
    make_env,
    milestones=[
        (0,      dict(n_enemies=4, enemy_move_every=7)),
        (50_000, dict(n_enemies=6, enemy_move_every=5)),
        (100_000,dict(n_enemies=8, enemy_move_every=3)),
    ],
    verbose=1,
)

print("✔ Using PyTorch device: auto (GPU if available)")
model = PPO(
    "MlpPolicy",                     # 21×21×3 is small; MLP suffices
    env,
    learning_rate=3e-4,
    batch_size=1024,
    tensorboard_log="logs",
    verbose=1,
    device="auto",
    policy_kwargs=dict(net_arch=[256, 256]),
)

model.learn(total_timesteps=TOTAL_STEPS, callback=[ckpt_cb, curric_cb])
model.save("runs/ppo_turret_final")
print("✓ Training finished – weights in runs/")