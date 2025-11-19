"""
Walk-forward backtesting pipeline for MicrostructureEnv + PPO (Transformer features).
"""

import os
import json
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.utils.seeds import set_global_seed
from src.data.loader import load_all_symbols
from src.data.features import (
    prepare_feature_dict,
    fit_scalers_on_train_dict,
    transform_dict_with_scalers,
    align_all_assets,
)
from src.envs.microstruct_env import MicrostructureEnv
from src.utils.metrics import evaluate_on_env, save_metrics_json
from src.agents.ppo_runner import TransformerObsEncoder


def make_env_from_slice(slice_dict, window, exec_params, seed):
    """
    Safe environment factory for SB3 VecEnv.
    """
    def _build():
        env = MicrostructureEnv(
            data_dict=slice_dict,
            window=window,
            exec_params=exec_params,
        )
        env.reset(seed=seed)
        return env
    return _build


def walkforward_run(
    out_dir=".",
    n_splits=6,
    train_window_steps=24 * 90,
    val_window_steps=24 * 30,
    test_window_steps=24 * 30,
    seed=42,
    ppo_timesteps_per_fold=100_000,
    window=168,
    exec_params=None,
):
    """
    Walk-forward training & evaluation of PPO with Transformer features.
    """
    set_global_seed(seed)

    raw = load_all_symbols()
    feat = prepare_feature_dict(raw)

    # We assume all assets have similar timestamp ranges and equal lengths.
    symbols = list(feat.keys())
    first_sym = symbols[0]
    df0 = feat[first_sym]
    T = len(df0)

    required = train_window_steps + val_window_steps + test_window_steps
    if T < required:
        raise RuntimeError(f"Dataset too short: T={T}, need at least {required}")

    max_start = T - required
    step = max(1, int(max_start / max(1, n_splits - 1)))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_dir = os.path.join(out_dir, "results", f"walkforward_{ts}")
    models_dir = os.path.join(out_dir, "models", f"walkforward_{ts}")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    fold_metrics = []
    fold_i = 0
    start_idx = 0

    # Preconfigured fast exec for speed
    exec_config = {
        "slice_participation": 0.02,
        "noise_scale": 0.0,
        "eta_temp": 1e-8,
        "gamma_perm": 1e-9,
    }
    if exec_params:
        exec_config.update(exec_params)

    policy_kwargs = dict(
        features_extractor_class=TransformerObsEncoder,
        features_extractor_kwargs=dict(
            d_model=128,
            nhead=4,
            num_layers=2,
            dropout=0.1,
        ),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )

    while True:
        train_start = start_idx
        train_end = train_start + train_window_steps
        val_end = train_end + val_window_steps
        test_end = val_end + test_window_steps

        if test_end > T:
            break

        # Slice raw features by index for all assets
        train_slice = {s: df.iloc[train_start:train_end].reset_index(drop=True)
                       for s, df in feat.items()}
        val_slice = {s: df.iloc[train_end:val_end].reset_index(drop=True)
                     for s, df in feat.items()}
        test_slice = {s: df.iloc[val_end:test_end].reset_index(drop=True)
                      for s, df in feat.items()}

        # Align timestamps per fold to avoid misalignment errors
        train_slice = align_all_assets(train_slice)
        val_slice = align_all_assets(val_slice)
        test_slice = align_all_assets(test_slice)

        # Fit scalers on train only
        scalers = fit_scalers_on_train_dict(train_slice, save=False)
        train_scaled = transform_dict_with_scalers(train_slice, scalers)
        val_scaled = transform_dict_with_scalers(val_slice, scalers)
        test_scaled = transform_dict_with_scalers(test_slice, scalers)

        # Build envs
        train_env = DummyVecEnv([make_env_from_slice(train_scaled, window, exec_config, seed)])
        model = PPO("MlpPolicy", train_env, verbose=0, policy_kwargs=policy_kwargs)
        model.learn(total_timesteps=ppo_timesteps_per_fold)

        val_env = DummyVecEnv([make_env_from_slice(val_scaled, window, exec_config, seed + 1)])
        test_env = DummyVecEnv([make_env_from_slice(test_scaled, window, exec_config, seed + 2)])

        val_metrics = evaluate_on_env(model, val_env)
        test_metrics = evaluate_on_env(model, test_env)

        result = {
            "fold": fold_i,
            "train_range": [train_start, train_end],
            "val_range": [train_end, val_end],
            "test_range": [val_end, test_end],
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "seed": seed,
            "timesteps": ppo_timesteps_per_fold,
        }

        save_metrics_json(result, os.path.join(results_dir, f"fold_{fold_i:02d}.json"))
        fold_metrics.append(result)

        # Move window
        fold_i += 1
        start_idx += step
        if fold_i >= n_splits:
            break

    save_metrics_json({"folds": fold_metrics}, os.path.join(results_dir, "summary.json"))
    print("Walk-forward complete →", results_dir)
    return results_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=".")
    parser.add_argument("--splits", type=int, default=6)
    parser.add_argument("--train_steps", type=int, default=24 * 90)
    parser.add_argument("--val_steps", type=int, default=24 * 30)
    parser.add_argument("--test_steps", type=int, default=24 * 30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ppo_steps", type=int, default=100_000)
    parser.add_argument("--window", type=int, default=168)
    args = parser.parse_args()

    walkforward_run(
        out_dir=args.out,
        n_splits=args.splits,
        train_window_steps=args.train_steps,
        val_window_steps=args.val_steps,
        test_window_steps=args.test_steps,
        seed=args.seed,
        ppo_timesteps_per_fold=args.ppo_steps,
        window=args.window,
    )
