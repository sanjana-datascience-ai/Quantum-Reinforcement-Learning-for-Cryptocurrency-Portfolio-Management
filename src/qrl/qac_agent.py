"""
Quantum Actor–Critic Agent (QAC)

Architecture:
- TransformerObsEncoder (same as PPO, but *frozen* here)
- QuantumFeatureEncoder: angle embedding
- VQCPolicy (PennyLane circuit on CPU, here default 12 qubits)
- Actor head: Dirichlet over assets: long-only allocation weights (sum=1)
- Critic head: V(s)

Training:
- A2C-style update with discounted returns
- Only quantum + actor/critic get gradients 
"""

from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.ppo_runner import TransformerObsEncoder
from src.qrl.q_encoder import QuantumFeatureEncoder
from src.qrl.vqc_policy import VQCPolicy


class QACAgent(nn.Module):
    def __init__(
        self,
        observation_space,
        action_dim: int,
        n_qubits: int = 12,
        n_layers: int = 1,
        lr: float = 2e-4,
    ):
        super().__init__()

        # Force CPU usage for QRL
        self.device = torch.device("cpu")

        # -------------------------
        # Classical encoder (frozen)
        # -------------------------
        self.encoder = TransformerObsEncoder(observation_space)
        latent_dim = self.encoder.features_dim

        # Freeze encoder parameters to save memory + compute
        for p in self.encoder.parameters():
            p.requires_grad = False

        # -------------------------
        # Quantum pipeline (trainable)
        # -------------------------
        self.q_encoder = QuantumFeatureEncoder(latent_dim, n_qubits)
        self.vqc = VQCPolicy(n_qubits=n_qubits, n_layers=n_layers)

        # -------------------------
        # Actor: Dirichlet over assets (α > 0)
        # -------------------------
        self.actor = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softplus(),  # ensures α_i > 0
        )

        # -------------------------
        # Critic: scalar V(s)
        # -------------------------
        self.critic = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Optimizer over *trainable* parts (q_encoder, vqc, actor, critic)
        self.optimizer = optim.Adam(
            list(self.q_encoder.parameters())
            + list(self.vqc.parameters())
            + list(self.actor.parameters())
            + list(self.critic.parameters()),
            lr=lr,
        )

        self.to(self.device)

    # ================================
    # INTERNAL ENCODER PIPELINE
    # ================================
    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs shape: (B, A, C, T) on CPU
        returns: (B, n_qubits)
        """
        obs = obs.to(self.device)

        # no_grad through Transformer encoder (frozen)
        with torch.no_grad():
            latent = self.encoder(obs)  # (B, latent_dim)

        latent = latent.detach()

        # Quantum feature encoder + VQC are trainable
        angles = self.q_encoder(latent)   # (B, n_qubits)
        q_out = self.vqc(angles)          # (B, n_qubits)

        return q_out

    # ================================
    # ACTOR
    # ================================
    def act(self, obs: torch.Tensor):
        """
        obs: (B, A, C, T)
        returns:
          - action: (B, action_dim) long-only simplex (Dirichlet sample)
          - logp: (B,)
          - value: (B,)
        """
        q_out = self._encode(obs)          # (B, n_qubits)
        alpha = self.actor(q_out) + 1e-4   # α_i > 0

        dist = torch.distributions.Dirichlet(alpha)
        action = dist.sample()
        logp = dist.log_prob(action)

        value = self.critic(q_out).squeeze(-1)

        return action, logp, value

    # ================================
    # VALUE-ONLY
    # ================================
    def value_fn(self, obs: torch.Tensor) -> torch.Tensor:
        q_out = self._encode(obs)
        return self.critic(q_out).squeeze(-1)

    # ================================
    # TRAINING UPDATE (A2C)
    # ================================
    def update(self, traj: Dict[str, Any], gamma: float = 0.995):
        """
        traj:
            {
                "logp":   [Tensor scalar per step, ...],
                "value":  [Tensor scalar per step, ...],
                "reward": [float, ...],
            }

        We only keep scalar tensors per step, so memory stays small
        even for long episodes (but our runner also caps max episode length).
        """

        logps = torch.stack(traj["logp"]).to(self.device)
        values = torch.stack(traj["value"]).to(self.device)
        rewards = torch.tensor(traj["reward"], dtype=torch.float32, device=self.device)

        # Discounted returns G_t
        G = []
        ret = 0.0
        for r in reversed(rewards.tolist()):
            ret = r + gamma * ret
            G.insert(0, ret)
        G = torch.tensor(G, dtype=torch.float32, device=self.device)

        # Advantage
        advantages = G - values.detach()

        # Losses
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
