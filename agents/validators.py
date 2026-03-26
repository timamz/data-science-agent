import csv
import os
import logging

logger = logging.getLogger("Validators")


def validate_csv(path, required_columns=None):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    with open(path) as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"CSV file is empty: {path}")
    if required_columns:
        missing = set(required_columns) - set(header)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
    logger.info(f"Validated CSV: {path} ({len(header)} columns)")
    return header


def validate_predictions(preds, expected_length):
    if preds is None:
        raise ValueError("Predictions are None")
    if len(preds) < expected_length:
        raise ValueError(
            f"Predictions length {len(preds)} < expected {expected_length}"
        )
    for i, p in enumerate(preds[:10]):
        try:
            float(p)
        except (TypeError, ValueError):
            raise ValueError(f"Prediction at index {i} is not numeric: {p}")


def validate_agent_output(result, required_keys):
    if not isinstance(result, dict):
        raise TypeError(f"Agent output must be dict, got {type(result).__name__}")
    missing = set(required_keys) - set(result.keys())
    if missing:
        raise KeyError(f"Agent output missing required keys: {missing}")
