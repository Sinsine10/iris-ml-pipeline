import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def load_training_table(csv_path: Path, target_col: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Missing target column {target_col!r}. Found: {list(df.columns)}")

    y = df[target_col]
    feature_cols = [c for c in df.columns if c != target_col]
    if not feature_cols:
        raise ValueError("No feature columns after removing the target column.")

    X = df[feature_cols]
    # Fail fast on obvious issues common in real CSVs
    if X.isna().any().any():
        raise ValueError(
            "Input contains missing values in feature columns. "
            "Impute or drop rows before training."
        )
    if y.isna().any():
        raise ValueError("Target column contains missing values.")

    return X, y, feature_cols


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Iris/tabular classifier from CSV.")
    parser.add_argument(
        "--data",
        type=Path,
        default=_repo_root() / "data" / "raw" / "iris.csv",
        help="Path to training CSV (default: data/raw/iris.csv)",
    )
    parser.add_argument(
        "--target",
        default="species",
        help="Name of the label column in the CSV (default: species)",
    )
    args = parser.parse_args()

    csv_path = args.data.resolve()
    if not csv_path.exists():
        raise SystemExit(
            f"Data file not found: {csv_path}\n"
            "Generate the sample CSV with: python scripts/create_iris_csv.py"
        )

    X, y_raw, feature_names = load_training_table(csv_path, args.target)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X.to_numpy(),
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=200, random_state=42),
            ),
        ]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    target_names = label_encoder.classes_
    acc = accuracy_score(y_test, y_pred)
    print(f"Data: {csv_path}")
    print(f"Features ({len(feature_names)}): {feature_names}")
    print(f"Test accuracy: {acc:.3f}\n")
    print(classification_report(y_test, y_pred, target_names=target_names))

    out_dir = _repo_root() / "artifacts"
    out_dir.mkdir(exist_ok=True)
    artifact = out_dir / "iris_pipeline.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "target_names": target_names,
            "target_column": args.target,
            "train_path": str(csv_path),
        },
        artifact,
    )
    print(f"Saved bundle to {artifact}")


if __name__ == "__main__":
    main()
