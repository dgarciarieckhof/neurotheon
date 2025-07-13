"""
Train a PPO agent on TurretEnv with curriculum, VecNormalize and GPU auto-selection.
Checkpoints every 10k steps are written to runs/.
"""

import os
import argparse
import torch
from pathlib import Path
from envs.turret_env import TurretEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from agents.curriculum import AdaptiveCurriculum
from agents.callback_vecnorm import SaveVecNormCallback
from agents.callback_kills import KillCountCallback
from agents.callback_stats import StatsCallback
from agents.callback_csv import EpisodeCSVCallback


# Training Configuration
CONFIG = {
    'total_steps': 500_000,
    'ckpt_freq': 50_000,
    'workers': 4,
    'seed': 42,
    'tensorboard_log': "logs",
    'save_dir': "runs",
    'eval_freq': 10_000,
    'eval_episodes': 10,
}

# PPO Hyperparameters
PPO_CONFIG = {
    'n_steps': 1024,
    'batch_size': 4096,
    'learning_rate': 1e-4,
    'gamma': 0.995,  # Slightly higher for longer-term planning
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,  # Encourage exploration
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'policy_kwargs': {
        'net_arch': [256, 256],
        'activation_fn': torch.nn.ReLU,
        'ortho_init': True,
    },
}

# Curriculum Configuration
CURRICULUM_MILESTONES = [
    # Phase 1 – Basic aim + low threat
    (0, {
        'n_enemies': 3,
        'enemy_move_every': 8,
        'path_probs': (1.0, 0.0, 0.0),  # straight only
        'cooldown_steps': 0,
        'sensor_min': 5,
    }),

    # Phase 2 – More targets, some movement variety
    (100_000, {
        'n_enemies': 5,
        'enemy_move_every': 6,
        'path_probs': (0.7, 0.2, 0.1),  # mostly straight, few zigzag
    }),

    # Phase 3 – Full mixed pathing, tighter timing
    (200_000, {
        'n_enemies': 7,
        'enemy_move_every': 5,
        'path_probs': (0.4, 0.35, 0.25),  # introduce erratic patterns
        'sensor_min': 3,
    }),

    # Phase 4 – Fire cooldown and reduced visibility
    (300_000, {
        'n_enemies': 9,
        'enemy_move_every': 4,
        'cooldown_steps': 1,
        'sensor_min': 3,
    }),

    # Phase 5 – Final test: tight, reactive combat
    (400_000, {
        'n_enemies': 11,
        'enemy_move_every': 3,
        'cooldown_steps': 1,
        'sensor_min': 2,
        'path_probs': (0.33, 0.33, 0.34),  # full pathing entropy
    }),
]


def make_env(**kw):
    """Create a TurretEnv instance with given parameters."""
    return TurretEnv(**kw)


def build_venv(config=None, eval_env=False):
    """Build vectorized environment with normalization."""
    if config is None:
        config = CONFIG
    
    # Default environment parameters
    default_params = {
        'n_enemies': 3,
        'enemy_move_every': 8,
        'path_probs': (1.0, 0.0, 0.0),
        'cooldown_steps': 0,
        'sensor_min': 5,
    }
    
    # For evaluation, use a fixed seed and single environment
    if eval_env:
        def _factory():
            return lambda: make_env(**default_params, seed=9999)
        base = SubprocVecEnv([_factory()])
    else:
        # Training environments with different seeds
        def _factory(idx: int):
            return lambda: make_env(**default_params, seed=config['seed'] + idx)
        base = SubprocVecEnv([_factory(i) for i in range(config['workers'])])
    
    # Normalize rewards but not observations (discrete spaces work better unnormalized)
    return VecNormalize(
        base, 
        norm_obs=False, 
        norm_reward=True, 
        clip_reward=10.0,
        gamma=PPO_CONFIG['gamma']
    )


def setup_callbacks(venv, config):
    """Set up all training callbacks."""
    save_freq_calls = config['ckpt_freq'] // config['workers']
    
    callbacks = []
    
    # Checkpointing
    ckpt_cb = CheckpointCallback(
        save_freq_calls, 
        config['save_dir'], 
        "ppo_turret_step",
        verbose=1
    )
    callbacks.append(ckpt_cb)
    
    # Curriculum learning
    curric_cb = AdaptiveCurriculum(
        make_env,
        milestones=CURRICULUM_MILESTONES,
        verbose=1,
    )
    callbacks.append(curric_cb)
    
    # VecNormalize saving
    vec_cb = SaveVecNormCallback(
        venv, 
        f"{config['save_dir']}/vecnorm.pkl", 
        save_freq_calls
    )
    callbacks.append(vec_cb)
    
    # Evaluation callback
    eval_env = build_venv(config, eval_env=True)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"{config['save_dir']}/best_model/",
        log_path=f"{config['save_dir']}/eval_logs/",
        eval_freq=config['eval_freq'] // config['workers'],
        n_eval_episodes=config['eval_episodes'],
        deterministic=True,
        verbose=1
    )
    callbacks.append(eval_cb)
    
    # Custom callbacks
    callbacks.extend([
        KillCountCallback(),
        StatsCallback(),
        EpisodeCSVCallback(),
    ])
    
    return callbacks


def create_model(venv, config):
    """Create PPO model with specified configuration."""
    return PPO(
        "MlpPolicy",
        venv,
        verbose=1,
        device="auto",
        tensorboard_log=config['tensorboard_log'],
        **PPO_CONFIG
    )


def resume_training(model_path, venv_path, config):
    """Resume training from a checkpoint."""
    print(f"Resuming training from {model_path}")
    
    # Load the model
    model = PPO.load(model_path, device="auto")
    
    # Create new environment (model will be set to this env)
    venv = build_venv(config)
    
    # Load normalization parameters if they exist
    if os.path.exists(venv_path):
        venv = VecNormalize.load(venv_path, venv)
        print(f"Loaded VecNormalize from {venv_path}")
    
    model.set_env(venv)
    return model, venv


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent on TurretEnv")
    parser.add_argument("--resume", type=str, help="Path to model checkpoint to resume from")
    parser.add_argument("--total-steps", type=int, default=CONFIG['total_steps'], 
                       help="Total training steps")
    parser.add_argument("--workers", type=int, default=CONFIG['workers'], 
                       help="Number of parallel workers")
    parser.add_argument("--seed", type=int, default=CONFIG['seed'], 
                       help="Random seed")
    parser.add_argument("--save-dir", type=str, default=CONFIG['save_dir'], 
                       help="Directory to save checkpoints")
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    config = CONFIG.copy()
    config.update({
        'total_steps': args.total_steps,
        'workers': args.workers,
        'seed': args.seed,
        'save_dir': args.save_dir,
    })
    
    # Create save directory
    Path(config['save_dir']).mkdir(parents=True, exist_ok=True)
    
    # Set random seed for reproducibility
    set_random_seed(config['seed'])
    
    # Print configuration
    print("Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("\nPPO Configuration:")
    for key, value in PPO_CONFIG.items():
        print(f"  {key}: {value}")
    print()
    
    # Build environment and model
    if args.resume:
        model_path = args.resume
        venv_path = f"{config['save_dir']}/vecnorm.pkl"
        model, venv = resume_training(model_path, venv_path, config)
    else:
        venv = build_venv(config)
        model = create_model(venv, config)
    
    # Setup callbacks
    callbacks = setup_callbacks(venv, config)
    
    # Train the model
    print("Starting training...")
    model.learn(
        total_timesteps=config['total_steps'],
        callback=callbacks,
        progress_bar=True,
    )
    
    # Save final model
    final_model_path = f"{config['save_dir']}/ppo_turret_final"
    final_venv_path = f"{config['save_dir']}/vecnorm_final.pkl"
    
    model.save(final_model_path)
    venv.save(final_venv_path)
    
    print(f"✓ Training finished!")
    print(f"  Final model saved to: {final_model_path}")
    print(f"  Final VecNormalize saved to: {final_venv_path}")
    print(f"  Best model saved to: {config['save_dir']}/best_model/")


def test_environment():
    """Test environment setup without training."""
    print("Testing environment setup...")
    
    # Test single environment
    env = make_env(seed=0)
    obs, _ = env.reset()
    print(f"Single env observation shape: {obs.shape}")
    
    # Test vectorized environment
    venv = build_venv({'workers': 2, 'seed': 0})
    obs = venv.reset()
    print(f"Vectorized env observation shape: {obs.shape}")
    
    # Test a few steps
    for i in range(5):
        actions = venv.action_space.sample()
        obs, rewards, dones, infos = venv.step(actions)
        print(f"Step {i+1}: rewards={rewards}, dones={dones}")
    
    venv.close()
    print("Environment test completed successfully!")


if __name__ == "__main__":
    from multiprocessing import set_start_method, freeze_support
    
    # Handle multiprocessing
    freeze_support()
    try:
        set_start_method("fork")
    except RuntimeError:
        pass
    
    # Add test mode
    import sys
    if "--test" in sys.argv:
        test_environment()
    else:
        main()