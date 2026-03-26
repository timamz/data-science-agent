from agents.base import BaseAgent
from agents.validators import validate_agent_output

SYSTEM = (
    "You are a model training agent. Generate Python code to train an ML model.\n\n"
    "IMPORTANT: Install ALL packages BEFORE importing them:\n"
    "  import subprocess, sys\n"
    "  subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', ...])\n\n"
    "The code receives: X_train, y_train, X_test (DataFrames), feature_names (list), time_budget_seconds (int).\n"
    "The code must define: val_predictions, test_predictions, val_true (arrays), model_name (str), model_params (dict).\n\n"
    "Split X_train/y_train into train/val (80/20, random_state=42).\n"
    "Use gradient boosting. NOT neural networks.\n\n"
    "Output only Python code in a ```python block."
)


class ModelAgent(BaseAgent):
    def run(self, X_train, y_train, X_test, feature_names, feedback="", time_budget=300):
        if X_train is None or y_train is None or X_test is None:
            raise ValueError("Input data cannot be None")

        summary = (
            f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}\n"
            f"Features: {feature_names}\n"
            f"Target: mean={y_train.mean():.2f}, std={y_train.std():.2f}, "
            f"min={y_train.min():.2f}, max={y_train.max():.2f}, "
            f"zero_frac={float((y_train == 0).mean()):.2f}\n"
            f"Time budget: {time_budget}s"
        )

        rag_context = self.kb.search("LGBMRegressor XGBRegressor fit early_stopping train_test_split")
        rag_str = "\n\n".join(rag_context[:3]) if rag_context else ""

        user = f"Data:\n{summary}"
        if rag_str:
            user += f"\n\nRelevant API docs:\n{rag_str}"
        if feedback:
            user += f"\n\nFeedback:\n{feedback}"

        response = self.call_llm(SYSTEM, user, temperature=0.5)
        code = self.extract_code(response) or ""
        self.logger.info(f"Generated training code:\n{code}")

        ns = self.execute_code(code, {
            "X_train": X_train, "y_train": y_train,
            "X_test": X_test, "feature_names": feature_names,
            "time_budget_seconds": time_budget,
        }, context=SYSTEM)

        result = {
            "val_predictions": ns["val_predictions"], "test_predictions": ns["test_predictions"],
            "val_true": ns["val_true"], "model_name": ns["model_name"],
            "model_params": ns["model_params"],
        }
        validate_agent_output(result, [
            "val_predictions", "test_predictions", "val_true", "model_name", "model_params",
        ])
        return result
