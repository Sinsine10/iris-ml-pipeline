# iris-ml-pipeline

Small end-to-end example: load the Iris dataset, fit a scaled logistic regression pipeline, report metrics, save a `joblib` bundle, and run a one-row prediction demo.

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

Artifacts are written to `artifacts/` (ignored by git).

## License

Add a `LICENSE` file when you publish the repository if you want an explicit license.
