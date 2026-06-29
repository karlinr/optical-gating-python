from graphlib import TopologicalSorter
from app.config import Config
from .estimators import estimator_registry

class PhaseManager:
    def __init__(self):
        self.estimators = {
            name: cls() for name, cls in estimator_registry.items()
        }

        self.source = Config.Gating.PHASE_SOURCE

        # Determine the full set of estimators to consider based on the enabled ones and their dependencies
        all_possible = set(Config.Gating.ENABLED_ESTIMATORS) | {self.source}
        resolved_all = set(all_possible)
        for name in all_possible:
            if name in self.estimators:
                for dep in getattr(self.estimators[name], "dependencies", []):
                    resolved_all.add(dep)

        # Compute execution order using topological sort based on dependencies
        self.execution_order = list(TopologicalSorter({
            name: getattr(self.estimators[name], "dependencies", []) for name in resolved_all
        }).static_order())


    def update(self, frame, timestamp) -> dict:
        # Determine active estimators based on enabled ones and their dependencies
        active_set = set(Config.Gating.ENABLED_ESTIMATORS) | {self.source}
        
        # Recursively add dependencies of active estimators to the active set
        for name in list(active_set):
            if name in self.estimators:
                for dep in self.estimators[name].active_dependencies:
                    active_set.add(dep)

        # Process estimators in the determined execution order, passing context from one to the next
        context = {}
        outputs = {}
        for name in self.execution_order:
            if name not in active_set:
                continue

            res = self.estimators[name].update(frame, timestamp=timestamp, context=context)
            outputs[name] = res
            if res is not None:
                context[name] = res

        # Construct the response for all enabled estimators, ensuring to include the active estimator's status and output
        response = {
            name: outputs.get(name) or {
                "phase": None, 
                "target_phase": None, 
                "barrier_phase": None, 
                "metrics": {}
            }
            for name in Config.Gating.ENABLED_ESTIMATORS
        }

        # Add the active estimator's status and output to the response, even if it's not in the enabled list
        active_estimator = self.estimators.get(self.source)
        is_ready = active_estimator.is_ready() if active_estimator else False
        active_output = outputs.get(self.source) if is_ready else {}

        response["ACTIVE"] = {
            "status": "READY" if is_ready else f"{self.source}_COLLECTING_FRAMES",
            "phase": active_output.get("phase"),
            "target_phase": active_output.get("target_phase"),
            "barrier_phase": active_output.get("barrier_phase"),
            "residual": active_output.get("residual"),
            "metrics": active_output.get("metrics", {})
        }
            
        return response