"""
Train a small classifier on the classic Iris dataset and save the model.
Run: python train_iris.py
"""

from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    X, y = load_iris(return_X_y=True)
    feature_names = load_iris().feature_names
    target_names = load_iris().target_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
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

    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.3f}\n")
    print(classification_report(y_test, y_pred, target_names=target_names))

    out_dir = Path(__file__).resolve().parent / "artifacts"
    out_dir.mkdir(exist_ok=True)
    artifact = out_dir / "iris_pipeline.joblib"
    joblib.dump(
        {"model": model, "feature_names": feature_names, "target_names": target_names},
        artifact,
    )
    print(f"Saved bundle to {artifact}")


if __name__ == "__main__":
    main()
