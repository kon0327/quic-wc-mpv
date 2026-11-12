"""QuIC-WC: compression of dense linear layers by compound (QuIC) orthogonal thin factors."""

from .config import QuICWCConfig, SideConfig
from .layers.linear import QuICWCLinear, prepare_for_inference

__all__ = ["QuICWCConfig", "SideConfig", "QuICWCLinear", "prepare_for_inference"]
__version__ = "0.1.0.dev0"
