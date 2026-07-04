"""Built-in indicator engines."""

from .base import Engine, EngineResult
from .gold_regime import GoldRegimeConfig, GoldRegimeEngine
from .iron_momentum import IronMomentumConfig, IronMomentumEngine
from .silver_flow import SilverFlowConfig, SilverFlowEngine

ENGINE_CLASSES = {
    "iron_momentum": (IronMomentumEngine, IronMomentumConfig),
    "silver_flow": (SilverFlowEngine, SilverFlowConfig),
    "gold_regime": (GoldRegimeEngine, GoldRegimeConfig),
}

__all__ = [
    "ENGINE_CLASSES",
    "Engine",
    "EngineResult",
    "GoldRegimeConfig",
    "GoldRegimeEngine",
    "IronMomentumConfig",
    "IronMomentumEngine",
    "SilverFlowConfig",
    "SilverFlowEngine",
]
