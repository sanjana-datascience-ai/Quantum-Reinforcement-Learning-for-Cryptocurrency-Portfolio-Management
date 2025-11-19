"""
Quantum RL for 3-Asset Crypto Portfolio Management
"""

import os
import gc
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import pennylane as qml

from src.utils.seeds import set_global_seed
from src.utils.metrics import evaluate_on_env, save_metrics_json
from src.data.loader import load_all_symbols
from src.data.features import (
    prepare_feature_dict,
    fit_scalers_on_train_dict,
    transform_dict_with_scalers,
    align_all_assets,
)
from src.envs.microstruct_env import MicrostructureEnv

# ============================================================
#  Hyper-parameters for QRL (good starting point)
# ============================================================

DEFAULT_ASSETS = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]  # 3 assets
DEFAULT_WINDOW = 72        # 3 days of hourly bars
DEFAULT_TOTAL_STEPS = 200_000
DEFAULT_GAMMA = 0.995
DEFAULT_LR = 1e-4
DEFAULT_N_QUBITS = 6       # safe on your laptop (can try 8 later)
DEFAULT_N_LAYERS = 2
DEFAULT_MAX_EP_LEN = 4096  # cap episode length to avoid memory blow-up


# ============================================================
#  Transformer-based Observation Encoder (same as PPO idea)
# ============================================================

class TransformerObsEncoder(nn.Module):
    """
    Same idea as in ppo_runner.py:

    obs: (B, A, C, T)
      - A = n_assets
      - C = channels (features + alloc)
      - T = window

    We treat time as sequence dimension and (asset, channel) as features.
    """

    def __init__(self, observation_space, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()

        if len(observation_space.shape) != 3:
            raise ValueError(
                f"[TransformerObsEncoder] Expected 3D obs, got {observation_space.shape}"
            )

        self.n_assets, self.n_channels, self.window = observation_space.shape
        self.d_model = d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=False,  # we use (T, B, E)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.input_proj = nn.Linear(self.n_channels, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

        self.features_dim = d_model

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        observations: (B, A, C, T)
        returns: (B, d_model)
        """
        x = observations  # (B, A, C, T)
        if x.dim() != 4:
            raise ValueError(
                f"[TransformerObsEncoder] Expected 4D input (B,A,C,T), got shape {x.shape}"
            )

        B, A, C, T = x.shape
        # (B, A, T, C)
        x = x.permute(0, 1, 3, 2)
        # (B*A, T, C)
        x = x.reshape(B * A, T, C)

        # Project channels -> d_model
        x = self.input_proj(x)  # (B*A, T, d_model)

        # Transformer expects (T, B*A, d_model)
        x = x.transpose(0, 1)  # (T, B*A, d_model)
        x = self.transformer(x)  # (T, B*A, d_model)

        # Mean pooling over time
        x = x.mean(dim=0)  # (B*A, d_model)
        x = self.layer_norm(x)

        # Reshape back to (B, A, d_model)
        x = x.view(B, A, self.d_model)
        # Global pooling over assets
        x = x.mean(dim=1)  # (B, d_model)

        return x


# ============================================================
#  Quantum Feature Encoder (classical → angles)
# ============================================================

class QuantumFeatureEncoder(nn.Module):
    """
    Latent (B, D) → angles (B, n_qubits) in [-π, π]
    """

    def __init__(self, in_features: int, n_qubits: int):
        super().__init__()
        self.in_features = in_features
        self.n_qubits = n_qubits

        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, n_qubits),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        return torch.pi * torch.tanh(raw)


# ============================================================
#  VQC Policy with PennyLane (default.qubit)
# ============================================================

class VQCPolicy(nn.Module):
    """
    PennyLane-based Variational Quantum Circuit:

      - n_qubits wires
      - AngleEmbedding over angles (one angle per qubit)
      - StronglyEntanglingLayers (n_layers)
      - Outputs expectation values <Z_i> for each qubit → R^n_qubits

    This is differentiable via PyTorch, but runs purely on CPU.
    """

    def __init__(self, n_qubits: int = 6, n_layers: int = 2):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # PennyLane device
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            # inputs: (n_qubits,)
            # weights: (n_layers, n_qubits, 3)
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        # TorchLayer automatically broadcasts over batch dim:
        #  angles: (B, n_qubits) → outputs: (B, n_qubits)
        self.vqc = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        return self.vqc(angles)


# ============================================================
#  Quantum Actor–Critic Agent
# ============================================================

class QACAgent(nn.Module):
    """
    Quantum Actor–Critic agent:

    obs (B, A, C, T)
      → TransformerObsEncoder (frozen)
      → QuantumFeatureEncoder (trainable)
      → VQCPolicy (trainable)
      → Actor head (Dirichlet over 3 assets)
      → Critic head V(s)

    We use an A2C-style update on full episodes.
    """

    def __init__(
        self,
        observation_space,
        action_dim: int,
        n_qubits: int = DEFAULT_N_QUBITS,
        n_layers: int = DEFAULT_N_LAYERS,
        lr: float = DEFAULT_LR,
    ):
        super().__init__()

        self.device = torch.device("cpu")

        # 1) Transformer encoder (frozen to save memory)
        self.encoder = TransformerObsEncoder(observation_space)
        latent_dim = self.encoder.features_dim

        for p in self.encoder.parameters():
            p.requires_grad = False

        # 2) Quantum pipeline
        self.q_encoder = QuantumFeatureEncoder(latent_dim, n_qubits)
        self.vqc = VQCPolicy(n_qubits=n_qubits, n_layers=n_layers)

        # 3) Actor: Dirichlet over assets
        self.actor = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softplus(),  # α_i > 0
        )

        # 4) Critic:
        self.critic = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Trainable params: quantum + actor + critic
        self.optimizer = optim.Adam(
            list(self.q_encoder.parameters())
            + list(self.vqc.parameters())
            + list(self.actor.parameters())
            + list(self.critic.parameters()),
            lr=lr,
        )

        self.to(self.device)

    # ---------- internal encoding ----------
    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, A, C, T)
        returns: (B, n_qubits)
        """
        obs = obs.to(self.device)

        with torch.no_grad():
            latent = self.encoder(obs)  # (B, latent_dim)

        latent = latent.detach()

        angles = self.q_encoder(latent)  # (B, n_qubits)
        q_out = self.vqc(angles)         # (B, n_qubits)

        return q_out

    # ---------- act ----------
    def act(self, obs: torch.Tensor):
        """
        returns:
          action: (B, action_dim) on simplex
          logp:   (B,)
          value:  (B,)
        """
        q_out = self._encode(obs)          # (B, n_qubits)
        alpha = self.actor(q_out) + 1e-4   # Dirichlet params

        dist = torch.distributions.Dirichlet(alpha)
        action = dist.sample()             # (B, action_dim)
        logp = dist.log_prob(action)       # (B,)

        value = self.critic(q_out).squeeze(-1)  # (B,)

        return action, logp, value

    def value_fn(self, obs: torch.Tensor) -> torch.Tensor:
        q_out = self._encode(obs)
        return self.critic(q_out).squeeze(-1)

    # ---------- A2C update ----------
    def update(self, traj: Dict[str, Any], gamma: float = DEFAULT_GAMMA):
        """
        traj:
            {
                "logp":   [Tensor scalar per step, ...],
                "value":  [Tensor scalar per step, ...],
                "reward": [float, ...],
            }
        """

        logps = torch.stack(traj["logp"]).to(self.device)   # (T,)
        values = torch.stack(traj["value"]).to(self.device) # (T,)
        rewards = torch.tensor(traj["reward"], dtype=torch.float32, device=self.device)

        # Discounted returns
        G_list: List[float] = []
        ret = 0.0
        for r in reversed(rewards.tolist()):
            ret = r + gamma * ret
            G_list.insert(0, ret)
        G = torch.tensor(G_list, dtype=torch.float32, device=self.device)  # (T,)

        advantages = G - values.detach()

        actor_loss = -(logps * advantages).mean()
        critic_loss = (G - values).pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
        }


# ============================================================
#  SB3-style model wrapper for evaluation_on_env()
# ============================================================

class QRLModelWrapper:
    """
    Minimal wrapper so src.utils.metrics.evaluate_on_env()
    can call model.predict(...) like SB3 PPO.
    """

    def __init__(self, agent: QACAgent, device: torch.device):
        self.agent = agent
        self.device = device

    def predict(self, obs, deterministic: bool = True):
        if isinstance(obs, np.ndarray):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        if obs_t.dim() == 3:
            obs_t = obs_t.unsqueeze(0)

        with torch.no_grad():
            q_out = self.agent._encode(obs_t)
            alpha = self.agent.actor(q_out) + 1e-4
            # Use mean of Dirichlet for deterministic action
            alloc = alpha / (alpha.sum(-1, keepdim=True) + 1e-12)

        return alloc.squeeze(0).cpu().numpy(), None


# ============================================================
#  Data / Env builders (3 assets)
# ============================================================

def build_three_asset_feature_dict(
    target_symbols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Load all *_features files, then restrict to 3 assets.
    """
    raw = load_all_symbols()            # dict[sym] -> df (features already)
    feat = prepare_feature_dict(raw)    # reloads *_1h_features.csv per symbol

    if target_symbols is None:
        target_symbols = DEFAULT_ASSETS

    # Filter features to requested assets (fallback to first 3 if missing)
    selected = {s: df for s, df in feat.items() if s in target_symbols}
    if len(selected) < 3:
        # fill up from whatever is available
        for s in feat.keys():
            if s not in selected and len(selected) < 3:
                selected[s] = feat[s]

    if len(selected) != 3:
        raise RuntimeError(f"Expected 3 assets, got {len(selected)}: {list(selected.keys())}")

    # Align timestamps
    data = align_all_assets(selected)
    return data


def build_train_val_envs(
    window: int = DEFAULT_WINDOW,
    split_ratio: float = 0.8,
    exec_params: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    target_symbols: Optional[List[str]] = None,
) -> Tuple[MicrostructureEnv, MicrostructureEnv, Dict[str, Any]]:

    data = build_three_asset_feature_dict(target_symbols=target_symbols)

    first_sym = list(data.keys())[0]
    n = len(data[first_sym])
    split_idx = int(n * split_ratio)

    train_raw = {s: df.iloc[:split_idx].reset_index(drop=True) for s, df in data.items()}
    val_raw = {s: df.iloc[split_idx:].reset_index(drop=True) for s, df in data.items()}

    scalers = fit_scalers_on_train_dict(train_raw, save=False)
    train_scaled = transform_dict_with_scalers(train_raw, scalers)
    val_scaled = transform_dict_with_scalers(val_raw, scalers)

    exec_cfg = {
        "slice_participation": 0.02,
        "noise_scale": 0.0,
        "eta_temp": 1e-8,
        "gamma_perm": 1e-9,
    }
    if exec_params:
        exec_cfg.update(exec_params)

    train_env = MicrostructureEnv(train_scaled, window=window, exec_params=exec_cfg)
    val_env = MicrostructureEnv(val_scaled, window=window, exec_params=exec_cfg)

    train_env.reset(seed=seed)
    val_env.reset(seed=seed + 1)

    return train_env, val_env, train_scaled


# ============================================================
#  Training loop
# ============================================================

def _format_hms(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def train_qrl_pipeline(
    out_dir: str = ".",
    total_steps: int = DEFAULT_TOTAL_STEPS,
    window: int = DEFAULT_WINDOW,
    split_ratio: float = 0.8,
    seed: int = 42,
    eval_freq: int = 25_000,
    lr: float = DEFAULT_LR,
    n_qubits: int = DEFAULT_N_QUBITS,
    n_layers: int = DEFAULT_N_LAYERS,
    gamma: float = DEFAULT_GAMMA,
    max_episode_len: int = DEFAULT_MAX_EP_LEN,
    target_symbols: Optional[List[str]] = None,
):

    # Output dirs
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(out_dir, "models", f"qrl_3asset_{ts}")
    results_dir = os.path.join(out_dir, "results", f"qrl_3asset_{ts}")
    logs_dir = os.path.join(out_dir, "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    log_path = os.path.join(logs_dir, f"qrl_3asset_{ts}.log")

    def log(msg: str):
        msg = str(msg)
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(f"[QRL-3ASSET] Logging → {log_path}")

    # Seed & device
    set_global_seed(seed)
    device = torch.device("cpu")
    log("[QRL-3ASSET] Using device: CPU")

    # Build envs
    train_env, val_env, train_scaled = build_train_val_envs(
        window=window,
        split_ratio=split_ratio,
        seed=seed,
        target_symbols=target_symbols,
    )

    action_dim = len(train_scaled)  # n_assets = 3
    log(f"[QRL-3ASSET] Assets = {list(train_scaled.keys())}")
    log(f"[QRL-3ASSET] n_assets = {action_dim}")

    # Agent
    agent = QACAgent(
        observation_space=train_env.observation_space,
        action_dim=action_dim,
        n_qubits=n_qubits,
        n_layers=n_layers,
        lr=lr,
    ).to(device)

    wrapper = QRLModelWrapper(agent, device)

    log("[QRL-3ASSET] Agent initialized.")
    log(f"[QRL-3ASSET] Training for {total_steps:,} steps...")
    log(f"[QRL-3ASSET] max_episode_len = {max_episode_len}")

    steps_done = 0
    next_eval = eval_freq
    episode_idx = 0
    start_t = time.time()

    recent_times: List[float] = []

    # ---------------- Main training loop ----------------
    while steps_done < total_steps:

        obs, _ = train_env.reset(seed=seed + episode_idx)
        done = False

        traj_logp: List[torch.Tensor] = []
        traj_val: List[torch.Tensor] = []
        traj_rew: List[float] = []

        ep_reward = 0.0
        ep_steps = 0

        while (
            not done
            and steps_done < total_steps
            and ep_steps < max_episode_len
        ):

            t0 = time.time()

            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            action, logp, value = agent.act(obs_t)
            act_np = action.squeeze(0).detach().cpu().numpy()

            nxt, reward, terminated, truncated, info = train_env.step(act_np)
            done = terminated or truncated
            ep_steps += 1

            traj_logp.append(logp.squeeze())
            traj_val.append(value.squeeze())
            traj_rew.append(float(reward))

            ep_reward += float(reward)
            obs = nxt
            steps_done += 1

            dt = time.time() - t0
            recent_times.append(dt)
            if len(recent_times) > 400:
                recent_times.pop(0)

            # Periodic GC
            if steps_done % 200 == 0:
                gc.collect()

            # Logging
            if steps_done % 500 == 0 or done or steps_done >= total_steps:
                elapsed = time.time() - start_t
                avg = np.mean(recent_times) if recent_times else 0.0
                rem = max(0, total_steps - steps_done)
                eta = rem * avg

                progress = steps_done / total_steps
                filled = int(progress * 30)
                bar = "█" * filled + "-" * (30 - filled)

                log(
                    f"[TRAIN] {steps_done:,}/{total_steps:,} "
                    f"({progress*100:4.1f}%) | {bar}\n"
                    f"        Ep {episode_idx}  Steps {ep_steps}  Reward {ep_reward:.2f}\n"
                    f"        Elapsed { _format_hms(elapsed) } | ETA { _format_hms(eta) }"
                )

        # ---- Update on this episode ----
        stats = agent.update(
            {"logp": traj_logp, "value": traj_val, "reward": traj_rew},
            gamma=gamma,
        )

        log(
            f"[EP {episode_idx}] steps={steps_done:,} "
            f"ep_steps={ep_steps}  R={ep_reward:.3f}  "
            f"loss={stats['loss']:.4f} A={stats['actor_loss']:.4f} C={stats['critic_loss']:.4f}"
        )

        # Free memory
        del traj_logp, traj_val, traj_rew
        gc.collect()

        # ---- Validation ----
        if steps_done >= next_eval:
            next_eval += eval_freq

            val = evaluate_on_env(wrapper, val_env)
            val["step"] = steps_done
            save_metrics_json(val, os.path.join(results_dir, f"val_{steps_done}.json"))

            log(
                f"[VAL] step={steps_done:,} "
                f"Sharpe={val.get('sharpe',0):.3f} "
                f"CumRet={val.get('cumulative_return',0):.3f}"
            )

        episode_idx += 1

    # Final save
    torch.save(agent.state_dict(), os.path.join(model_dir, "final_qrl_agent.pt"))
    log(f"[QRL-3ASSET] Training completed. Model saved to {model_dir}")

    final = evaluate_on_env(wrapper, val_env)
    save_metrics_json(final, os.path.join(results_dir, "final.json"))
    log(
        f"[FINAL] Sharpe={final.get('sharpe',0):.3f} "
        f"CumRet={final.get('cumulative_return',0):.3f}"
    )

    return model_dir, results_dir, final


# ============================================================
#  CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default=".")
    p.add_argument("--steps", type=int, default=DEFAULT_TOTAL_STEPS)
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    p.add_argument("--split", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_freq", type=int, default=25_000)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--n_qubits", type=int, default=DEFAULT_N_QUBITS)
    p.add_argument("--n_layers", type=int, default=DEFAULT_N_LAYERS)
    p.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    p.add_argument("--max_ep_len", type=int, default=DEFAULT_MAX_EP_LEN)
    p.add_argument(
        "--assets",
        type=str,
        nargs="*",
        default=DEFAULT_ASSETS,
        help="3 asset symbols, e.g. BTC/USDT ETH/USDT BNB/USDT",
    )

    args = p.parse_args()

    train_qrl_pipeline(
        out_dir=args.out,
        total_steps=args.steps,
        window=args.window,
        split_ratio=args.split,
        seed=args.seed,
        eval_freq=args.eval_freq,
        lr=args.lr,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        gamma=args.gamma,
        max_episode_len=args.max_ep_len,
        target_symbols=args.assets,
    )
