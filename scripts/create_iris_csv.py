"""
Regenerate data/raw/iris.csv from sklearn (same values as the canonical Iris set).
Run from repo root: python scripts/create_iris_csv.py
"""

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "data" / "raw" / "iris.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    iris = load_iris(as_frame=True)
    frame = iris.frame
    df = frame.drop(columns=["target"]).rename(
        columns={
            "sepal length (cm)": "sepal_length_cm",
            "sepal width (cm)": "sepal_width_cm",
            "petal length (cm)": "petal_length_cm",
            "petal width (cm)": "petal_width_cm",
        }
    )
    df["species"] = iris.target_names[frame["target"].to_numpy()]
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
