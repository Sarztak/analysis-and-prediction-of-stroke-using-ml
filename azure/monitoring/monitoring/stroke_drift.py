import argparse
import json
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

def main():
    # -----------------------------------------------
    # Parse command-line arguments passed by Azure ML
    # -----------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_data", type=str, required=True,
                        help="Path to the baseline dataset (original test data).")
    parser.add_argument("--modified_data", type=str, required=True,
                        help="Path to the modified dataset used to simulate data drift.")
    parser.add_argument("--drift_report", type=str, required=True,
                        help="Path where the output JSON drift report will be written.")
    args = parser.parse_args()

    print("Baseline dataset path:", args.baseline_data)
    print("Modified dataset path:", args.modified_data)

    # -----------------------------------------------------
    # Load both datasets from the provided file paths.
    # In Azure ML, these paths point to mounted blob storage.
    # -----------------------------------------------------
    base = pd.read_csv(args.baseline_data)
    mod = pd.read_csv(args.modified_data)

    # -----------------------------------------------------
    # Remove non-feature columns:
    # - "id" is an identifier and should not be used for drift.
    # - "stroke" is the target (label) column.
    # -----------------------------------------------------
    target_col = "stroke"
    drop_cols = ["id", target_col]

    # Keep only predictor columns (features)
    base_x = base.drop(columns=[c for c in drop_cols if c in base.columns])
    mod_x = mod.drop(columns=[c for c in drop_cols if c in mod.columns])

    # -----------------------------------------------------
    # Select only numeric feature columns, since KS test
    # applies to continuous distributions.
    # -----------------------------------------------------
    numeric_cols = base_x.select_dtypes(include=[np.number]).columns.tolist()
    print("Numeric columns used for drift detection:", numeric_cols)

    drift_results = []

    # -----------------------------------------------------
    # Compute drift for each numeric feature using
    # Kolmogorov–Smirnov (KS) test.
    # -----------------------------------------------------
    for col in numeric_cols:
        base_vals = base_x[col].dropna()
        mod_vals = mod_x[col].dropna()

        # Skip if either dataset has no valid values
        if len(base_vals) == 0 or len(mod_vals) == 0:
            continue

        # KS statistic measures distance between distributions
        ks_stat, p_val = ks_2samp(base_vals, mod_vals)

        drift_results.append(
            {
                "feature": col,
                "baseline_mean": float(base_vals.mean()),
                "modified_mean": float(mod_vals.mean()),
                "ks_stat": float(ks_stat),
                "p_value": float(p_val),
            }
        )

    # -----------------------------------------------------
    # Sort results by KS statistic (largest drift first)
    # -----------------------------------------------------
    drift_results = sorted(drift_results, key=lambda x: x["ks_stat"], reverse=True)

    # -----------------------------------------------------
    # Create a summary dictionary to be stored as JSON
    # -----------------------------------------------------
    summary = {
        "num_features": len(drift_results),
        "num_features_ks>0.1": int(sum(r["ks_stat"] > 0.1 for r in drift_results)),
        "drift_results": drift_results,
    }

    print("==== Drift Summary ====")
    print(json.dumps(summary, indent=2))

    # -----------------------------------------------------
    # Write the drift report to the output path specified
    # by Azure ML component output binding.
    # -----------------------------------------------------
    with open(args.drift_report, "w") as f:
        json.dump(summary, f)

    print(f"Drift report successfully written to: {args.drift_report}")

if __name__ == "__main__":
    main()
