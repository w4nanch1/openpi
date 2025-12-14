from __future__ import annotations

from openpi.models_pytorch.steering_methods.base import ActivationProcessor


def create_activation_processor(steering_config: dict | None) -> ActivationProcessor | None:
    """Create an activation processor based on steering config.
    """
    if steering_config is None:
        return None
    
    method = steering_config.get("method", "none")
    params = steering_config.get("params", {})
    
    if method == "none":
        return None
    
    # Import and instantiate specific steering methods
    if method == "caa":
        # TODO: Import and create CAA processor when implemented
        # from openpi.models_pytorch.steering_methods.caa import CAAProcessor
        # return CAAProcessor(**params)
        raise NotImplementedError(f"Steering method '{method}' not yet implemented")
    
    raise ValueError(f"Unknown steering method: {method}")

