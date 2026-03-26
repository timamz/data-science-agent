import csv
import io

from agents.base import BaseAgent
from agents.validators import validate_csv, validate_agent_output

SYSTEM = (
    "You are a data preprocessing agent. Generate Python code to preprocess a dataset.\n\n"
    "IMPORTANT: Install ALL packages BEFORE importing them:\n"
    "  import subprocess, sys\n"
    "  subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'pandas', 'numpy', 'scikit-learn'])\n\n"
    "The code receives: train_path, test_path (file path strings).\n"
    "The code must define: X_train (numeric DataFrame), y_train (Series), X_test (numeric DataFrame), feature_names (list).\n\n"
    "Rules:\n"
    "- ONLY preprocess. Do NOT train models.\n"
    "- All columns must be numeric (encode categoricals).\n"
    "- Handle missing values.\n"
    "- Do NOT scale features.\n\n"
    "Output only Python code in a ```python block."
)


class DataAgent(BaseAgent):
    def run(self, train_path, test_path, task_description, feedback=""):
        validate_csv(train_path)
        validate_csv(test_path)

        profile = self._build_profile(train_path)

        rag_context = self.kb.search("preprocessing LabelEncoder fillna read_csv feature engineering")
        rag_str = "\n\n".join(rag_context[:3]) if rag_context else ""

        user = f"Task:\n{task_description}\n\nData Profile:\n{profile}"
        if rag_str:
            user += f"\n\nRelevant API docs:\n{rag_str}"
        if feedback:
            user += f"\n\nFeedback:\n{feedback}"

        response = self.call_llm(SYSTEM, user)
        code = self.extract_code(response) or ""
        self.logger.info(f"Generated preprocessing code:\n{code}")

        ns = self.execute_code(
            code, {"train_path": train_path, "test_path": test_path}, context=SYSTEM,
        )
        result = {
            "X_train": ns["X_train"], "y_train": ns["y_train"],
            "X_test": ns["X_test"], "feature_names": ns["feature_names"],
        }
        validate_agent_output(result, ["X_train", "y_train", "X_test", "feature_names"])
        return result

    def _build_profile(self, path):
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
            rows, total = [], 0
            nulls = {h: 0 for h in header}
            for row in reader:
                total += 1
                if total <= 5:
                    rows.append(row)
                for h, v in zip(header, row):
                    if v.strip() == "":
                        nulls[h] += 1

        buf = io.StringIO()
        buf.write(f"Columns ({len(header)}): {', '.join(header)}\n")
        buf.write(f"Total rows: {total}\n\n")
        missing = {h: c for h, c in nulls.items() if c > 0}
        if missing:
            buf.write("Missing values:\n")
            for h, c in missing.items():
                buf.write(f"  {h}: {c}/{total}\n")
            buf.write("\n")
        buf.write("Sample rows:\n")
        for i, row in enumerate(rows):
            buf.write(f"Row {i}: " + " | ".join(f"{h}={v}" for h, v in zip(header, row)) + "\n")
        return buf.getvalue()
