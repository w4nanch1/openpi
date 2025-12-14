from __future__ import annotations

from typing import Protocol

import torch


class ActivationProcessor(Protocol):
    """Activation post-processing/steering interface (e.g., CAA).

    Implement this protocol elsewhere and register on the model. The hook will
    call `steer_activation` for each captured activation.
    """

    def steer_activation(self, layer_idx, activation, context):
        """Handle a captured activation.

        Args:
            layer_idx: Layer index where the activation was captured.
            activation: Tensor of shape [batch, seq_len, hidden_size] (or similar).
            context: Optional metadata (e.g., batch info) provided by caller.
        """
        ...


