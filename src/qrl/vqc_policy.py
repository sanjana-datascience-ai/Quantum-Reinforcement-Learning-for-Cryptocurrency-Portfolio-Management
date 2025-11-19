"""
VQCPolicy

Parametrized quantum circuit wrapped as a Torch nn.Module.

- Uses PennyLane's default.qubit device (CPU)
- AngleEmbedding over n_qubits
- StronglyEntanglingLayers with correct weight shape (L, n_qubits, 3)
- Outputs expectation values <Z_i> for each qubit → R^n_qubits
"""

from typing import Tuple

import pennylane as qml
import torch
import torch.nn as nn


class VQCPolicy(nn.Module):
    def __init__(self, n_qubits: int = 12, n_layers: int = 1):
        """
        n_qubits: number of qubits (expressivity). 12 is a good target for
                  demonstrating potential quantum advantage while remaining
                  feasible on CPU.
        n_layers: depth of StronglyEntanglingLayers. We keep it small (1)
                  to control compute and memory.
        """
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # PennyLane CPU device
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            """
            inputs:  (n_qubits,) angles
            weights: (n_layers, n_qubits, 3)
            """
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}

        # TorchLayer handles batching automatically:
        # inputs: (B, n_qubits) → outputs: (B, n_qubits)
        self.vqc = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        """
        angles: (batch, n_qubits)
        returns: (batch, n_qubits) expectation values
        """
        return self.vqc(angles)
