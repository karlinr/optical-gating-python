import importlib
import pkgutil
from pathlib import Path
from .base import estimator_registry, PhaseEstimator, register_estimator

# Automatically discover and dynamically import all Python files in this folder
package_dir = str(Path(__file__).parent)
for _, module_name, _ in pkgutil.walk_packages([package_dir]):
    if module_name == "base":
        continue
    importlib.import_module(f"{__name__}.{module_name}")

__all__ = ["estimator_registry", "PhaseEstimator", "register_estimator"]