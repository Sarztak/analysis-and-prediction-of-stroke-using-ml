# monitoring/evaluate_model.py
# Evaluate H2O/MLflow model on a CSV test dataset.
# Expected arguments:
#   --model_dir: local folder that contains an MLflow model (folder with MLmodel)
#   --test_data: CSV file with a "stroke" target column
#   --metrics_output: path to write a JSON file with metrics

import argparse
import json
import os

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)


def parse_args():
    """Parse command-line arguments passed by Azure ML component."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Local directory that contains the MLflow model (folder with MLmodel).",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Path to CSV test data.",
    )
    parser.add_argument(
        "--metrics_output",
        type=str,
        required=True,
        help="Path to write metrics JSON.",
    )
    return parser.parse_args()


def find_mlflow_model_dir(base_dir: str) -> str:
    """Find the directory that actually contains MLmodel.

    Sometimes Azure ML passes a higher-level folder. We search inside it
    for a directory that contains the MLmodel file.
    """
    # Case 1: base_dir itself contains MLmodel
    if os.path.isfile(os.path.join(base_dir, "MLmodel")):
        return base_dir

    # Case 2: search recursively
    for root, dirs, files in os.walk(base_dir):
        if "MLmodel" in files:
            return root

    return None


def main():
    args = parse_args()
    print(f"[INFO] test_data path: {args.test_data}")
    print(f"[INFO] model_dir input: {args.model_dir}")

    # -----------------------------
    # 1. Basic path validations
    # -----------------------------
    if not os.path.exists(args.test_data):
        raise FileNotFoundError(f"Test CSV not found: {args.test_data}")

    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Model base directory not found: {args.model_dir}")

    # -----------------------------
    # 2. Load test data
    # -----------------------------
    df = pd.read_csv(args.test_data)

    target_col = "stroke"
    drop_cols = ["id", target_col]

    y_true = df[target_col]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    print(f"[INFO] Loaded test data, shape = {df.shape}")

    # -----------------------------
    # 3. Locate MLflow model folder
    # -----------------------------
    model_base = args.model_dir
    real_model_dir = find_mlflow_model_dir(model_base)

    if real_model_dir is None:
        raise FileNotFoundError(
            f"Could not find 'MLmodel' under: {model_base}. "
            f"Please check that stroke_best_model_xgb_final/model_azure "
            f"is included in the component 'code' snapshot."
        )

    print(f"[INFO] Resolved MLflow model directory: {real_model_dir}")

    # -----------------------------
    # 4. Initialize H2O (for h2o flavor)
    # -----------------------------
    try:
        import h2o

        print("[INFO] Initializing H2O runtime...")
        h2o.init(
            strict_version_check=False,
            max_mem_size="3G",
            nthreads=-1,
            log_level="WARN",
        )
        h2o.no_progress()
    except Exception as e:
        # Not fatal if it still works, but we log it
        print(f"[WARN] Could not initialize H2O: {e}")

    # -----------------------------
    # 5. Load MLflow model
    # -----------------------------
    print("[INFO] Loading MLflow model via mlflow.pyfunc.load_model(...)")
    model = mlflow.pyfunc.load_model(real_model_dir)
    print(f"[INFO] Loaded model object: {model} (type={type(model)})")

    if model is None:
        # Fail loudly if model didn't load correctly
        raise RuntimeError(f"mlflow.pyfunc.load_model('{real_model_dir}') returned None")

    # -----------------------------
    # 6. Predict
    # -----------------------------
    print("[INFO] Running predictions...")
    raw_pred = model.predict(X)
    print(f"[INFO] Raw prediction type: {type(raw_pred)}")

    # ---- 把輸出整理成「stroke 機率 (p1)」 ----
    if isinstance(raw_pred, pd.DataFrame):
        # H2O / MLflow typical columns: ["predict", "p0", "p1"]
        if "p1" in raw_pred.columns:
            y_prob = raw_pred["p1"].values
            print("[INFO] Using 'p1' column as stroke probability.")
        elif "predict" in raw_pred.columns:
            # Fallback: if only predict exists, treat 0/1 as probabilities
            y_prob = raw_pred["predict"].values.astype(float)
            print(
                "[WARN] 'p1' column not found; using 'predict' as probability (0/1)."
            )
        else:
            raise ValueError(
                f"Unknown prediction columns: {raw_pred.columns.tolist()}"
            )
    else:
        # Fallback: array-like predictions
        arr = np.asarray(raw_pred)
        if arr.ndim > 1 and arr.shape[1] >= 2:
            # If it really is [p0, p1], take column 1
            y_prob = arr[:, 1]
            print("[INFO] Using column index 1 as stroke probability.")
        else:
            y_prob = arr
            print("[INFO] Using 1D array as stroke probability.")

    # Apply your chosen threshold (from notebook tuning)
    threshold = 0.43
    y_pred = (y_prob >= threshold).astype(int)
    print(f"[INFO] Using threshold = {threshold} to binarize predictions.")

    # -----------------------------
    # 7. Compute metrics
    # -----------------------------
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception as e:
        print(f"[WARN] Cannot compute AUC: {e}")
        auc = None

    metrics = {
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "auc": float(auc) if auc is not None else None,
        "n_samples": int(len(df)),
    }

    print("[INFO] Evaluation metrics:")
    print(json.dumps(metrics, indent=2))

    # -----------------------------
    # 8. Save metrics JSON
    # -----------------------------
    metrics_dir = os.path.dirname(args.metrics_output)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)

    with open(args.metrics_output, "w") as f:
        json.dump(metrics, f)

    print(f"[INFO] Metrics written to {args.metrics_output}")


if __name__ == "__main__":
    main()
