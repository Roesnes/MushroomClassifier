"""Train a mushroom edibility classifier and export the model artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import Any, cast

RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a mushroom edibility classifier.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("secondary_data.csv"),
        help="Path to the secondary mushroom dataset (semicolon separated).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory where the trained model and reports will be stored.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=20,
        help="Number of parameter combinations to sample during hyper-parameter search.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path.resolve()}")

    df = pd.read_csv(
        path,
        sep=";",
        na_values=["?", "", " "],
    )
    return df


def build_pipeline(feature_names: list[str]) -> Pipeline:
    numeric_features = ["cap-diameter", "stem-height", "stem-width"]
    categorical_features = [col for col in feature_names if col not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    classifier = RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return pipeline


def hyperparameter_search(
    pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, n_iter: int
) -> tuple[Pipeline, dict[str, Any], dict[str, Any]]:
    param_distributions = {
        "classifier__n_estimators": [100, 200, 400, 600],
        "classifier__max_depth": [None, 10, 20, 40],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2", 0.4],
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=3,
        random_state=RANDOM_STATE,
        verbose=1,
        n_jobs=1,
    )
    search.fit(X_train, y_train)
    best_model: Any = search.best_estimator_
    if not isinstance(best_model, Pipeline):
        raise TypeError("Best estimator from search is not a Pipeline instance.")

    best_params = cast(dict[str, Any], search.best_params_)
    cv_results = cast(dict[str, Any], search.cv_results_)

    return best_model, cv_results, best_params


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, output_dir: Path) -> dict[str, float]:
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["edible", "poisonous"], output_dict=True)

    y_test_numeric = (y_test == "poisonous").astype(int)
    y_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test_numeric, y_proba)

    report_typed = cast(dict[str, Any], report)

    metrics = {
        "precision_poisonous": float(report_typed["poisonous"]["precision"]),
        "recall_poisonous": float(report_typed["poisonous"]["recall"]),
        "f1_poisonous": float(report_typed["poisonous"]["f1-score"]),
        "precision_edible": float(report_typed["edible"]["precision"]),
        "recall_edible": float(report_typed["edible"]["recall"]),
        "f1_edible": float(report_typed["edible"]["f1-score"]),
        "accuracy": float(report_typed["accuracy"]),
        "macro_avg_f1": float(report_typed["macro avg"]["f1-score"]),
        "roc_auc": float(roc_auc),
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save classification report as JSON for traceability.
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    # Save a confusion matrix plot to visually inspect model performance.
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["edible", "poisonous"], ax=ax)
    ax.set_title("Confusion Matrix - Mushroom Classifier")
    fig.tight_layout()
    fig.savefig(str(output_dir / "confusion_matrix.png"), dpi=200)
    plt.close(fig)

    # Persist a human-readable report as well.
    report_text = cast(str, classification_report(y_test, y_pred, target_names=["edible", "poisonous"]))
    with (output_dir / "classification_report.txt").open("w", encoding="utf-8") as fp:
        fp.write(report_text)

    return metrics


def save_feature_importances(model: Pipeline, output_dir: Path) -> None:
    classifier = model.named_steps.get("classifier")
    preprocessor = model.named_steps.get("preprocessor")

    if classifier is None or preprocessor is None:
        return

    if not hasattr(classifier, "feature_importances_"):
        return

    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(output_dir / "feature_importances.csv", index=False)


def save_search_results(cv_results: dict[str, Any], best_params: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cv_results_df = pd.DataFrame(cv_results)
    cv_results_df.to_csv(output_dir / "cv_results.csv", index=False)

    with (output_dir / "best_params.json").open("w", encoding="utf-8") as fp:
        json.dump(best_params, fp, indent=2)


def main() -> None:
    args = parse_args()

    df = load_dataset(args.data_path)

    X = df.drop(columns=["class"])
    y = df["class"].map({"e": "edible", "p": "poisonous"})

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    pipeline = build_pipeline(feature_names)
    model, cv_results, best_params = hyperparameter_search(pipeline, X_train, y_train, args.n_iter)

    metrics = evaluate_model(model, X_test, y_test, args.output_dir)
    save_feature_importances(model, args.output_dir)
    save_search_results(cv_results, best_params, args.output_dir)

    model_path = args.output_dir / "mushroom_classifier.joblib"
    joblib.dump(model, model_path)

    summary_lines = [
        "Mushroom classifier training complete.",
        f"Best model saved to: {model_path}",
        "Key metrics:",
    ]
    summary_lines.extend([f"  - {metric}: {value:.4f}" for metric, value in metrics.items()])

    summary_lines.append("Best hyper-parameters:")
    summary_lines.extend([f"  - {key}: {value}" for key, value in best_params.items()])

    with (args.output_dir / "training_summary.txt").open("w", encoding="utf-8") as fp:
        fp.write("\n".join(summary_lines))

    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
