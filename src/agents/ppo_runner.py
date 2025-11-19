"""
PPO BASELINE for Quantum RL Comparison
Optimized for MicrostructureEnv + 3-asset crypto portfolio management
"""

import os
import json
from datetime import datetime

import numpy as np
import torch as th
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.utils.seeds import set_global_seed
from src.utils.metrics import evaluate_on_env, save_metrics_json, metrics_to_csv

from src.data.loader import load_all_symbols
from src.data.features import (
    prepare_feature_dict,
    fit_scalers_on_train_dict,
    transform_dict_with_scalers,
    align_all_assets,
)

from src.envs.microstruct_env import MicrostructureEnv

DEFAULT_TOTAL_TIMESTEPS = 200_000
EVAL_FREQ = 20_000
DEFAULT_WINDOW = 72          # 3 days of hourly data
DEFAULT_N_ASSETS = 3         # fair to QRL (3-asset env)


# ============================================================
#  Lightweight Transformer Encoder (Optimized)
# ============================================================

class LightTransformerEncoder(BaseFeaturesExtractor):
    """
    A lighter Transformer-based encoder for 3D observations:
        obs shape: (A, C, T)
    """

    def __init__(self, observation_space, d_model=64, nhead=2, num_layers=1):
        super().__init__(observation_space, features_dim=d_model)

        A, C, T = observation_space.shape
        self.A = A
        self.C = C
        self.T = T
        self.d_model = d_model

        self.input_proj = nn.Linear(C, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2 * d_model,
            dropout=0.05,
            batch_first=False,
        )

        self.tf = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, obs: th.Tensor):
        """
        obs: (B, A, C, T)
        """
        B, A, C, T = obs.shape
        x = obs.permute(0, 1, 3, 2)  # (B, A, T, C)
        x = x.reshape(B * A, T, C)

        x = self.input_proj(x)
        x = x.transpose(0, 1)        # (T, B*A, d_model)
        x = self.tf(x)               # (T, B*A, d_model)
        x = x.mean(0)                # (B*A, d_model)
        x = self.layer_norm(x)

        # pool over assets
        return x.view(B, A, self.d_model).mean(1)


# ============================================================
#  Utilities
# ============================================================

def _select_first_n_assets(data_dict, n):
    """
    Deterministically select the first N assets (alphabetically).
    PPO and QRL must use the SAME subset for a fair comparison.
    """
    syms = sorted(data_dict.keys())
    chosen = syms if n is None or n <= 0 or n >= len(syms) else syms[:n]
    print(f"[ppo] Using {len(chosen)} assets: {chosen}")
    return {s: data_dict[s] for s in chosen}


def make_env(data_slice, window, exec_params, reward_params, seed=None):
    """
    Closure that builds a fresh MicrostructureEnv for DummyVecEnv.
    """
    def _build():
        env = MicrostructureEnv(
            data_dict=data_slice,
            window=window,
            exec_params=exec_params,
            reward_params=reward_params,
            risk_window=24 * 7,
            max_drawdown=0.8,
        )
        env.reset(seed=seed)
        return env
    return _build


def build_train_val_envs(
    data,
    window,
    split_ratio,
    exec_params,
    reward_params,
    seed,
):
    """
    Train/val split, per-split scalers, VecNormalize wrappers.
    """

    first = list(data.keys())[0]
    n = len(data[first])
    split = int(n * split_ratio)

    train_raw = {s: df.iloc[:split].reset_index(drop=True) for s, df in data.items()}
    val_raw   = {s: df.iloc[split:].reset_index(drop=True) for s, df in data.items()}

    # Fit scalers on TRAIN only (no leakage)
    scalers = fit_scalers_on_train_dict(train_raw, save=False)
    train = transform_dict_with_scalers(train_raw, scalers)
    val   = transform_dict_with_scalers(val_raw, scalers)

    # Build vectorized envs
    train_env = DummyVecEnv([make_env(train, window, exec_params, reward_params, seed)])
    val_env   = DummyVecEnv([make_env(val,   window, exec_params, reward_params, seed + 1)])

    # Normalization (critical for stable PPO learning)
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )
    # For val: normalize obs using running stats, but DO NOT normalize reward
    val_env = VecNormalize(
        val_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )

    train_env.reset()
    val_env.reset()

    return train_env, val_env


# ============================================================
#  Main Training Loop
# ============================================================

def train_ppo_pipeline(
    out_dir=".",
    total_timesteps=DEFAULT_TOTAL_TIMESTEPS,
    window=DEFAULT_WINDOW,
    split_ratio=0.8,
    seed=42,
    eval_freq=EVAL_FREQ,
    lr=1e-4,
    batch_size=128,
    n_assets=DEFAULT_N_ASSETS,
    exec_params=None,
):
    """
    Strong PPO baseline:
      - 3-asset MicrostructureEnv
      - Obs + reward normalization
      - Lightweight Transformer encoder
      - Tuned PPO hyperparameters
    """

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_dir = os.path.join(out_dir, "models", f"ppo_{ts}")
    results_dir = os.path.join(out_dir, "results", f"ppo_{ts}")
    logdir = os.path.join(out_dir, "logs", f"ppo_{ts}")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    set_global_seed(seed)

    # ---------- Load data ----------
    raw = load_all_symbols()          # dict[sym] -> *_1h_features.csv based
    feat = prepare_feature_dict(raw)  # reload features from disk
    aligned = align_all_assets(feat)  # align timestamps
    aligned = _select_first_n_assets(aligned, n_assets)

    # ---------- Execution & reward configs ----------
    exec_cfg = {
        "slice_participation": 0.02,
        "noise_scale": 0.0,
        "eta_temp": 1e-8,
        "gamma_perm": 1e-9,
    }
    if exec_params:
        exec_cfg.update(exec_params)

    reward_cfg = {
        "lambda_vol": 2.0,   # volatility penalty
        "eta_turn":  0.05,   # turnover penalty
        "gamma_cvar": 1.0,   # tail-risk penalty
    }

    # ---------- Envs ----------
    train_env, val_env = build_train_val_envs(
        aligned,
        window,
        split_ratio,
        exec_cfg,
        reward_cfg,
        seed,
    )

    # ---------- Policy ----------
    policy_kwargs = dict(
        features_extractor_class=LightTransformerEncoder,
        features_extractor_kwargs=dict(
            d_model=64,
            nhead=2,
            num_layers=1,
        ),
        net_arch=dict(pi=[64, 32], vf=[64, 32]),
    )

    # ---------- PPO model ----------
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=lr,
        batch_size=batch_size,
        n_steps=512,
        gamma=0.98,
        gae_lambda=0.92,
        ent_coef=0.01,
        clip_range=0.15,
        max_grad_norm=0.5,
        tensorboard_log=logdir,
        verbose=1,
        policy_kwargs=policy_kwargs,
    )

    # ---------- Checkpoint callback ----------
    cb = CheckpointCallback(
        save_freq=eval_freq,
        save_path=model_dir,
        name_prefix="ppo_ckpt",
    )

    t_done = 0
    best_sharpe = -1e9
    metrics_hist = []

    # ====================================================
    #  Training & periodic evaluation
    # ====================================================
    while t_done < total_timesteps:
        to_learn = min(eval_freq, total_timesteps - t_done)

        model.learn(
            total_timesteps=to_learn,
            reset_num_timesteps=False,
            callback=cb,
        )
        t_done += to_learn

        # Evaluate on validation env
        val_metrics = evaluate_on_env(model, val_env)
        val_metrics["timesteps"] = t_done
        val_metrics["timestamp"] = datetime.now().isoformat()
        val_metrics["algo"] = "ppo"
        val_metrics["window"] = window
        val_metrics["n_assets"] = len(aligned)

        metrics_hist.append(val_metrics)

        save_metrics_json(
            val_metrics,
            os.path.join(results_dir, f"val_{t_done}.json"),
        )

        sharpe = float(val_metrics.get("sharpe", 0.0))
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            model.save(os.path.join(model_dir, "best_model.zip"))

        print(
            f"[PPO] Timesteps={t_done:,} | Sharpe={sharpe:.3f} | "
            f"CumRet={val_metrics.get('cumulative_return', 0.0):.3f}"
        )

    # ---------- Final evaluation ----------
    final = evaluate_on_env(model, val_env)
    final["timesteps"] = t_done
    final["timestamp"] = datetime.now().isoformat()
    final["algo"] = "ppo"
    final["window"] = window
    final["n_assets"] = len(aligned)

    save_metrics_json(final, os.path.join(results_dir, "final_metrics.json"))
    model.save(os.path.join(model_dir, "final_model.zip"))
    metrics_to_csv(metrics_hist, os.path.join(results_dir, "history.csv"))

    print("\n[PPO] Training complete.")
    print("Model dir:", model_dir)
    print("Results dir:", results_dir)
    print(
        f"FINAL: Sharpe={final.get('sharpe',0.0):.3f}, "
        f"CumRet={final.get('cumulative_return',0.0):.3f}"
    )

    return model_dir, results_dir, final


# ============================================================
#  CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=".")
    parser.add_argument("--steps", type=int, default=DEFAULT_TOTAL_TIMESTEPS)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_freq", type=int, default=EVAL_FREQ)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--n_assets", type=int, default=DEFAULT_N_ASSETS)
    args = parser.parse_args()

    train_ppo_pipeline(
        out_dir=args.out,
        total_timesteps=args.steps,
        window=args.window,
        split_ratio=args.split,
        seed=args.seed,
        eval_freq=args.eval_freq,
        lr=args.lr,
        batch_size=args.batch,
        n_assets=args.n_assets,
    )
