"""CLI helper to run inference with the trained mushroom classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

DEFAULT_MODEL_PATH = Path("models/mushroom_classifier.joblib")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict mushroom edibility from feature values.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the serialized model produced by train_model.py.",
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="Path to a JSON file containing a dictionary of feature values.",
    )
    return parser.parse_args()


def load_features(path: Path) -> pd.DataFrame:
    data: dict[str, Any]
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    if isinstance(data, dict):
        return pd.DataFrame([data])
    if isinstance(data, list):
        return pd.DataFrame(data)

    raise ValueError("Input JSON must be an object or a list of objects.")


def main() -> None:
    args = parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found at {args.model_path}")

    model = joblib.load(args.model_path)
    features = load_features(args.input_json)

    predictions = model.predict(features)
    probabilities = model.predict_proba(features)

    results = []
    for i, label in enumerate(predictions):
        proba = probabilities[i]
        results.append(
            {
                "prediction": label,
                "probability_edible": float(proba[0]),
                "probability_poisonous": float(proba[1]),
            }
        )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
