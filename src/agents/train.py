"""
Train a PPO agent on TurretEnv with curriculum, VecNormalize and GPU auto-selection.
Checkpoints every 10k steps are written to runs/.
"""

from envs.turret_env import TurretEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from agents.curriculum import CurriculumCallback
from agents.callback_kills import KillCountCallback
from agents.callback_vecnorm import SaveVecNormCallback

TOTAL_STEPS = 200_000
CKPT_FREQ   = 10_000


# ── env factory ──────────────────────────────────────────────────
def make_env(**kw):
    return TurretEnv(**kw)


# ── build VecNormalize env (rewards only) ────────────────────────
base_vec = DummyVecEnv([lambda: make_env(n_enemies=2, enemy_move_every=10)])
venv     = VecNormalize(base_vec, norm_obs=False,
                        norm_reward=True, clip_reward=10.0)

# ── callbacks ────────────────────────────────────────────────────
ckpt_cb = CheckpointCallback(
    save_freq=CKPT_FREQ,
    save_path="runs",
    name_prefix="ppo_turret_step"
)

curric_cb = CurriculumCallback(
    make_env,
    milestones=[
        (0,       dict(n_enemies=2, enemy_move_every=10)),
        (50_000,  dict(n_enemies=4, enemy_move_every=8)),
        (100_000, dict(n_enemies=6, enemy_move_every=6)),
        (150_000, dict(n_enemies=8, enemy_move_every=4)),
    ],
    verbose=1,
)

vec_cb = SaveVecNormCallback(venv, "runs/vecnorm.pkl", CKPT_FREQ)

print("✔ Using PyTorch device: auto (GPU if available)")

model = PPO(
    "MlpPolicy",             # 21×21×3 → MLP is fine
    venv,
    learning_rate=3e-4,
    batch_size=1024,
    tensorboard_log="logs",
    verbose=1,
    device="auto",
    policy_kwargs=dict(net_arch=[256, 256]),
)

model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=[ckpt_cb, curric_cb, KillCountCallback(), vec_cb]
)

# save both weights and VecNormalize statistics
model.save("runs/ppo_turret_final")
venv.save("runs/vecnorm.pkl")

print("✓ Training finished – weights + vecnorm saved in runs/")