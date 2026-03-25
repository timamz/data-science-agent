from agents.base import BaseAgent

SYSTEM = (
    "You are an evaluation agent. Analyze model metrics and provide feedback.\n"
    "Respond with JSON: {\"analysis\": \"...\", \"feedback\": \"...\", \"should_continue\": bool}\n"
    "Be specific: suggest concrete changes. Only output JSON."
)


class EvalAgent(BaseAgent):
    def run(self, val_true, val_preds, model_name, params, iteration, history, task_description):
        metrics = self._compute_metrics(val_true, val_preds)
        self.logger.info(f"Iteration {iteration}: {metrics}")

        history_str = "\n".join(
            f"  Iter {h['iteration']}: {h['model']} MSE={h['mse']:.2f}" for h in history
        ) if history else "None"

        response = self.call_llm(SYSTEM, (
            f"Task:\n{task_description}\n\nIteration: {iteration}\n"
            f"Model: {model_name}\nParams: {params}\nMetrics: {metrics}\n"
            f"History:\n{history_str}"
        ))
        parsed = self.parse_json(response, {
            "analysis": "", "feedback": "Try different approach", "should_continue": True,
        })

        return {
            **metrics,
            "feedback": f"Metrics: {metrics}\nAnalysis: {parsed.get('analysis', '')}\n"
                        f"Suggestions: {parsed.get('feedback', '')}",
            "should_continue": parsed.get("should_continue", True),
        }

    def _compute_metrics(self, val_true, val_preds):
        true = [float(v) for v in val_true]
        pred = [float(v) for v in val_preds]
        n = len(true)
        mse = sum((t - p) ** 2 for t, p in zip(true, pred)) / n
        mae = sum(abs(t - p) for t, p in zip(true, pred)) / n
        mean_y = sum(true) / n
        ss_tot = sum((t - mean_y) ** 2 for t in true)
        r2 = 1 - mse * n / ss_tot if ss_tot > 0 else 0.0
        return {"mse": round(mse, 2), "rmse": round(mse ** 0.5, 2),
                "mae": round(mae, 2), "r2": round(r2, 4)}
