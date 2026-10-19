"""
CerebroSpheroid Root Compatibility Wrapper
Re-exports the core architecture from src.model.
"""

from src.model import (
    SensoryInputDNN,
    VolumetricBrainCore,
    MotorOutputDNN,
    CerebroSpheroid,
    build_cerebro_model,
)

__all__ = [
    "SensoryInputDNN",
    "VolumetricBrainCore",
    "MotorOutputDNN",
    "CerebroSpheroid",
    "build_cerebro_model",
]

if __name__ == "__main__":
    from src.model import main as model_main
    model_main()
