import json
import os
import time
import logging

logger = logging.getLogger("Tracker")


class _SafeEncoder(json.JSONEncoder):
    def default(self, o):
        try:
            import numpy as np
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
        except ImportError:
            pass
        return str(o)


class ExperimentTracker:
    def __init__(self, model_name, time_budget, lib_versions):
        self.run_id = time.strftime("%Y%m%d_%H%M%S")
        self.config = {
            "model": model_name,
            "time_budget": time_budget,
            "lib_versions": lib_versions,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.iterations = []
        self.agent_stats = {}
        logger.info(f"Experiment run {self.run_id} started")

    def log_iteration(self, iteration, model_name, params, metrics):
        entry = {
            "iteration": iteration,
            "model": model_name,
            "params": str(params),
            "metrics": metrics,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.iterations.append(entry)
        logger.info(f"Logged iteration {iteration}: MSE={metrics.get('mse')}")

    def log_agent_stats(self, agents):
        for agent in agents:
            self.agent_stats[agent.name] = {
                "calls": len(agent.log),
                "successes": sum(1 for entry in agent.log if entry["success"]),
                "avg_latency": round(
                    sum(entry["latency"] for entry in agent.log) / max(len(agent.log), 1), 2
                ),
                "total_tokens": getattr(agent, "total_tokens", 0),
            }

    def save(self):
        os.makedirs("benchmarking/runs", exist_ok=True)
        path = f"benchmarking/runs/{self.run_id}.json"
        data = {
            "run_id": self.run_id,
            "config": self.config,
            "iterations": self.iterations,
            "agent_stats": self.agent_stats,
            "best_mse": min(
                (it["metrics"]["mse"] for it in self.iterations if "mse" in it["metrics"]),
                default=None,
            ),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, cls=_SafeEncoder)
        logger.info(f"Experiment saved to {path}")
        return path
