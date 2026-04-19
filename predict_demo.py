"""
Load the saved Iris model and run one prediction (demo).
Run after train_iris.py: python predict_demo.py
"""

from pathlib import Path

import joblib
import numpy as np


def main() -> None:
    artifact = Path(__file__).resolve().parent / "artifacts" / "iris_pipeline.joblib"
    if not artifact.exists():
        raise SystemExit(f"Missing {artifact}. Run train_iris.py first.")

    bundle = joblib.load(artifact)
    model = bundle["model"]
    target_names = bundle["target_names"]
    feature_names = bundle["feature_names"]

    # Example: one flower with typical setosa-like measurements
    sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    pred = model.predict(sample)[0]
    proba = model.predict_proba(sample)[0]

    print("Features:", {name: float(val) for name, val in zip(feature_names, sample[0])})
    print("Predicted class:", target_names[pred])
    print("Class probabilities:")
    for name, p in zip(target_names, proba):
        print(f"  {name}: {p:.3f}")


if __name__ == "__main__":
    main()
