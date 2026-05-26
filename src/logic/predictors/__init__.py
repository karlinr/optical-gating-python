import importlib
import pkgutil
from pathlib import Path
from .base import predictor_registry, PhasePredictor, register_predictor

package_dir = str(Path(__file__).parent)
for _, module_name, _ in pkgutil.walk_packages([package_dir]):
    if module_name == "base":
        continue
    importlib.import_module(f"{__name__}.{module_name}")

__all__ = ["predictor_registry", "PhasePredictor", "register_predictor"]