import os
import csv
import sys
import time
import platform
import logging
import io
import contextlib

from agents.base import BaseAgent
from agents.data_agent import DataAgent
from agents.model_agent import ModelAgent
from agents.eval_agent import EvalAgent
from agents.guardrails import check_code_safety, get_restricted_builtins
from agents.validators import validate_predictions

SYSTEM = (
    "You are an autonomous data science agent. You solve ML tasks by writing and executing Python code.\n\n"
    "You work in a loop: you write a ```python code block, it gets executed, you see the output, then you reason and write more code.\n"
    "If you write no code block, the loop ends.\n\n"
    "You MUST use these specialized agent functions (they are AI agents that generate and execute code):\n"
    "- data = preprocess(train_path, test_path, task, feedback='') -> dict with X_train, y_train, X_test, feature_names\n"
    "- result = train_model(X_train, y_train, X_test, feature_names, feedback='', time_budget=300) -> dict with val_predictions, test_predictions, val_true, model_name, model_params\n"
    "- evaluation = evaluate(val_true, val_preds, model_name, params, iteration, history, task) -> dict with mse, rmse, mae, r2, feedback, should_continue\n\n"
    "Other available functions:\n"
    "- search_docs(query) -> list of relevant library documentation strings\n"
    "- save_submission(predictions) -> saves submission.csv\n"
    "- verify_submission() -> dict with test_mse, n_samples (checks against solution.csv)\n"
    "- get_elapsed() -> seconds since start\n"
    "- print() -> captured and shown to you\n\n"
    "IMPORTANT RULES:\n"
    "- You MUST call preprocess() for data preprocessing — do NOT write preprocessing code yourself\n"
    "- You MUST call train_model() for model training — do NOT write training code yourself\n"
    "- You MUST call evaluate() after each training run\n"
    "- You MUST call search_docs() before each major step to look up relevant API documentation\n"
    "  Example: search_docs('LGBMRegressor fit early_stopping') or search_docs('optuna create_study pruner')\n"
    "- You CAN write code for: analysis, ensembling, calling the above functions, checking results\n"
    "- Pass feedback from evaluate() to train_model() or preprocess() on the next iteration\n\n"
    "Workflow:\n"
    "1. Understand the task and plan your approach\n"
    "2. Call preprocess() to prepare data\n"
    "3. Call train_model() with the data, then evaluate()\n"
    "4. Based on evaluation feedback, iterate: call train_model() again with feedback, or re-preprocess\n"
    "5. Try ensembling predictions from multiple models\n"
    "6. When satisfied or time is running out, call save_submission() with best predictions\n"
    "7. Call verify_submission() to check the test score\n"
    "8. Stop by writing a response with NO code block"
)


class OrchestratorAgent(BaseAgent):
    def __init__(self):
        super().__init__("Orchestrator")
        self.data_agent = DataAgent("DataAgent", kb=self.kb)
        self.model_agent = ModelAgent("ModelAgent", kb=self.kb)
        self.eval_agent = EvalAgent("EvalAgent", kb=self.kb)
        self.start_time = None

    def run(self, time_budget=600):
        self.start_time = time.time()

        task = self._read_prompt()
        system_info = self._get_system_info()
        data_profile = self.data_agent._build_profile("data/train.csv")
        lib_versions = self._install_and_index_libs()

        plan_context = (
            f"=== TASK ===\n{task}\n\n"
            f"=== SYSTEM ===\n{system_info}\n\n"
            f"=== DATA PROFILE ===\n{data_profile}\n\n"
            f"=== INSTALLED LIBRARIES ===\n{lib_versions}\n\n"
            f"=== TIME BUDGET ===\n{time_budget} seconds total"
        )

        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": plan_context},
        ]

        namespace = self._build_namespace(task)

        self.logger.info("Starting orchestrator loop")

        while True:
            elapsed = time.time() - self.start_time
            if elapsed > time_budget:
                self.logger.info(f"Time budget exhausted ({elapsed:.0f}s)")
                self._auto_save(namespace)
                break

            response = self.call_llm_multi(messages, temperature=0.5)
            if not response:
                self.logger.warning("Empty LLM response, retrying once")
                response = self.call_llm_multi(messages, temperature=0.5)
                if not response:
                    self.logger.warning("Still empty, stopping")
                    self._auto_save(namespace)
                    break

            messages.append({"role": "assistant", "content": response})

            code = self.extract_code(response)
            if code is None:
                self.logger.info("No code block in response, agent is done")
                self._auto_save(namespace)
                break

            self.logger.info(f"Executing code step ({elapsed:.0f}s elapsed)")

            stdout_capture = io.StringIO()
            error_msg = None
            is_safe, violations = check_code_safety(code)
            if not is_safe:
                error_msg = f"Code blocked by guardrails: {', '.join(violations)}"
                self.logger.warning(error_msg)
            else:
                try:
                    with contextlib.redirect_stdout(stdout_capture):
                        exec(code, namespace)
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {e}"
                    self.logger.warning(f"Code execution error: {error_msg}")

            output = stdout_capture.getvalue()
            result_msg = ""
            if output:
                result_msg += f"Output:\n{output}\n"
            if error_msg:
                result_msg += f"Error:\n{error_msg}\n"
            if not result_msg:
                result_msg = "Code executed successfully (no output)."

            result_msg += f"\n[Elapsed: {time.time() - self.start_time:.0f}s / {time_budget}s]"
            messages.append({"role": "user", "content": result_msg})

        self._print_benchmark()

    def _read_prompt(self):
        with open("prompt.txt") as f:
            return f.read()

    def _get_system_info(self):
        try:
            import psutil
            ram = f"{psutil.virtual_memory().total / (1024**3):.1f} GB"
        except ImportError:
            ram = "unknown"

        return (
            f"Platform: {platform.machine()} ({platform.system()})\n"
            f"CPU cores: {os.cpu_count()}\n"
            f"RAM: {ram}\n"
            f"Python: {platform.python_version()}"
        )

    def _install_and_index_libs(self):
        import subprocess
        libs = ["pandas", "numpy", "scikit-learn", "lightgbm", "xgboost", "optuna"]
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q"] + libs,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        pkg_to_import = {
            "pandas": "pandas", "numpy": "numpy",
            "scikit-learn": "sklearn", "lightgbm": "lightgbm",
            "xgboost": "xgboost", "optuna": "optuna",
        }

        devnull = io.StringIO()
        for pkg, mod in pkg_to_import.items():
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                self.kb.index_library(mod)

        return self.kb.get_versions_summary()

    def _build_namespace(self, task):
        history = []
        all_results = []
        agent = self

        def preprocess(train_path="data/train.csv", test_path="data/test.csv",
                       task_desc=task, feedback=""):
            return agent.data_agent.run(train_path, test_path, task_desc, feedback)

        def train_model(X_train, y_train, X_test, feature_names,
                        feedback="", time_budget=300):
            result = agent.model_agent.run(
                X_train, y_train, X_test, feature_names, feedback, time_budget)
            all_results.append((None, result["test_predictions"]))
            return result

        def evaluate(val_true, val_preds, model_name, params,
                     iteration=None, hist=None, task_desc=task):
            it = iteration if iteration is not None else len(history) + 1
            h = hist if hist is not None else history
            result = agent.eval_agent.run(
                val_true, val_preds, model_name, params, it, h, task_desc)
            history.append({
                "iteration": it, "model": model_name,
                "mse": result["mse"], "rmse": result["rmse"],
                "mae": result["mae"], "r2": result["r2"],
            })
            if all_results and all_results[-1][0] is None:
                all_results[-1] = (result["mse"], all_results[-1][1])
            return result

        def search_docs(query):
            return agent.kb.search(query)

        def save_submission(predictions, sample_path="data/sample_submition.csv"):
            with open(sample_path) as f:
                reader = csv.reader(f)
                next(reader)
                indices = [int(row[0]) for row in reader]
            validate_predictions(predictions, max(indices) + 1)
            with open("submission.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["index", "prediction"])
                for idx in indices:
                    writer.writerow([idx, float(predictions[idx])])
            agent.logger.info("Submission saved")
            return "submission.csv saved"

        def get_elapsed():
            return round(time.time() - agent.start_time, 1)

        def verify_submission():
            preds = {}
            with open("submission.csv") as f:
                for row in csv.DictReader(f):
                    preds[int(row["index"])] = float(row["prediction"])
            truth = {}
            with open("data/solution.csv") as f:
                for row in csv.DictReader(f):
                    if row["Usage"] == "Public":
                        truth[int(row["index"])] = float(row["prediction"])
            common = sorted(set(preds) & set(truth))
            mse = sum((preds[i] - truth[i]) ** 2 for i in common) / len(common)
            return {"test_mse": round(mse, 2), "n_samples": len(common)}

        return {
            "__builtins__": get_restricted_builtins(),
            "preprocess": preprocess,
            "train_model": train_model,
            "evaluate": evaluate,
            "search_docs": search_docs,
            "save_submission": save_submission,
            "get_elapsed": get_elapsed,
            "verify_submission": verify_submission,
            "history": history,
            "all_results": all_results,
        }

    def _auto_save(self, namespace):
        if "best_test_predictions" in namespace and namespace["best_test_predictions"] is not None:
            try:
                namespace["save_submission"](namespace["best_test_predictions"])
                self.logger.info("Auto-saved best predictions")
                result = namespace["verify_submission"]()
                self.logger.info(f"Auto-save verification: {result}")
            except Exception as e:
                self.logger.warning(f"Auto-save failed: {e}")
        elif "all_results" in namespace and namespace["all_results"]:
            try:
                valid = [(m, p) for m, p in namespace["all_results"] if m is not None]
                if not valid:
                    self.logger.warning("No valid results to auto-save")
                    return
                best = min(valid, key=lambda x: x[0])
                namespace["save_submission"](best[1])
                self.logger.info("Auto-saved from all_results")
                result = namespace["verify_submission"]()
                self.logger.info(f"Auto-save verification: {result}")
            except Exception as e:
                self.logger.warning(f"Auto-save failed: {e}")

    def _print_benchmark(self):
        all_agents = [self, self.data_agent, self.model_agent, self.eval_agent]
        print("\n=== AGENT STATS ===")
        for a in all_agents:
            calls = len(a.log)
            ok = sum(1 for l in a.log if l["success"])
            avg = sum(l["latency"] for l in a.log) / max(calls, 1)
            print(f"  {a.name}: {calls} calls, {ok} ok, avg {avg:.1f}s")
        total = time.time() - self.start_time
        print(f"  Total time: {total:.0f}s")
