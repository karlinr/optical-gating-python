from app.config import Config
from .estimators import estimator_registry

class PhaseManager:
    def __init__(self):
        self.estimators = {
            name: cls() for name, cls in estimator_registry.items()
        }

    def update(self, frame, timestamp) -> dict:
        source = Config.Gating.PHASE_SOURCE.upper()
        base_to_run = set(Config.Gating.ENABLED_ESTIMATORS) | {source}

        # Resolve dependencies dynamically without hardcoding specific class logic
        resolved_to_run = set()
        for name in base_to_run:
            if name in self.estimators:
                resolved_to_run.add(name)
                for dep in self.estimators[name].active_dependencies:
                    resolved_to_run.add(dep)

        # Determine execution order based on dependency count
        execution_order = sorted(
            resolved_to_run,
            key=lambda n: len(getattr(self.estimators[n], "dependencies", []))
        )

        # Execute estimators sequentially and accumulate context
        context = {}
        outputs = {}
        for name in execution_order:
            res = self.estimators[name].update(frame, timestamp=timestamp, context=context)
            outputs[name] = res
            if res is not None:
                context[name] = res

        # Construct the response dictionary
        response = {
            name: outputs.get(name) or {"phase": None, "metrics": {}}
            for name in Config.Gating.ENABLED_ESTIMATORS
        }

        active_estimator = self.estimators.get(source)
        is_ready = active_estimator.is_ready() if active_estimator else False
        active_output = outputs.get(source) if is_ready else {}

        response["ACTIVE"] = {
            "status": "READY" if is_ready else f"{source}_COLLECTING_FRAMES",
            "phase": active_output.get("phase"),
            "target_phase": active_output.get("target_phase"),
            "barrier_phase": active_output.get("barrier_phase"),
            "metrics": active_output.get("metrics", {})
        }
            
        return response