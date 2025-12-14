from __future__ import annotations

import torch


class ActivationProcessor:
    """
    Base class for activation steering.
    """

    def steer_activation(self, layer_idx: int, activation: torch.Tensor, context: dict | None = None) -> None:
        """Handle a captured activation for steering.

        Args:
            layer_idx: Layer index where the activation was captured.
            activation: Tensor of shape [batch, seq_len, hidden_size] (or similar).
            context: Optional metadata (e.g., batch idx) provided by caller.
        """
        raise NotImplementedError("Subclasses must implement steer_activation")


