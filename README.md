# iris-ml-pipeline

Small end-to-end example in a **data-on-disk** layout: read a CSV from `data/raw/`, fit a scaled logistic regression pipeline, report metrics, save a `joblib` bundle, and run a one-row prediction demo.

## Data layout

| Path | Purpose |
|------|---------|
| `data/raw/iris.csv` | Versioned training table (snake_case columns + `species` label) |
| `scripts/create_iris_csv.py` | Regenerates `iris.csv` from sklearn if you need to refresh the file |

Replace `iris.csv` with your own file and pass `--data` / `--target` to `train_iris.py` (same numeric feature columns, string or categorical labels).

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\pip install -r requirements.txt
# macOS / Linux
# source .venv/bin/activate && pip install -r requirements.txt
```

Optional editable install (uses `pyproject.toml`):

```bash
pip install -e .
```

## Run

```bash
python train_iris.py
python predict_demo.py
```

Custom dataset:

```bash
python train_iris.py --data data/raw/your_table.csv --target class_name
```

Artifacts are written to `artifacts/` (ignored by git).

## License

Add a `LICENSE` file when you publish the repository if you want an explicit license.
