import json
import os
import logging

logger = logging.getLogger("Compare")

RUNS_DIR = "benchmarking/runs"


def load_runs():
    if not os.path.isdir(RUNS_DIR):
        return []
    runs = []
    for fname in sorted(os.listdir(RUNS_DIR)):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(RUNS_DIR, fname)) as f:
                    runs.append(json.load(f))
            except (json.JSONDecodeError, ValueError):
                logger.warning(f"Skipping corrupt file: {fname}")
    return runs


def compare_runs():
    runs = load_runs()
    if not runs:
        return "No runs found in benchmarking/runs/."
    lines = [
        f"{'Run ID':<20} | {'Model':<25} | {'Best MSE':<10} | {'Iters':<6} | Budget",
        "-" * 80,
    ]
    for run in sorted(runs, key=lambda r: r.get("best_mse") or float("inf")):
        rid = run["run_id"]
        model = run["config"].get("model", "?")[:25]
        best = run.get("best_mse")
        best_str = f"{best:.2f}" if best is not None else "N/A"
        iters = len(run.get("iterations", []))
        budget = run["config"].get("time_budget", "?")
        lines.append(f"{rid:<20} | {model:<25} | {best_str:<10} | {iters:<6} | {budget}s")
    return "\n".join(lines)


def compare_agents():
    runs = load_runs()
    if not runs:
        return "No runs found in benchmarking/runs/."
    lines = [
        f"{'Run ID':<20} | {'Agent':<14} | {'Calls':<6} | {'OK':<4} | {'Avg Lat':<8} | Tokens",
        "-" * 80,
    ]
    for run in runs:
        for name, stats in run.get("agent_stats", {}).items():
            lines.append(
                f"{run['run_id']:<20} | {name:<14} | {stats['calls']:<6} | "
                f"{stats['successes']:<4} | {stats['avg_latency']:<8.1f} | "
                f"{stats.get('total_tokens', 0)}"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    print(compare_runs())
    print()
    print(compare_agents())
