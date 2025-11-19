"""
Quantum Feature Encoder
It maps features → bounded quantum angles to be used by the VQCPolicy.

Architecture:
    latent (B, D) → MLP → (B, n_qubits) in [-π, π]
"""

import torch
import torch.nn as nn


class QuantumFeatureEncoder(nn.Module):
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
        """
        x: (batch, in_features)
        returns: (batch, n_qubits) angles in [-π, π]
        """
        raw = self.net(x)
        return torch.pi * torch.tanh(raw)
