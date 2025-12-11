import argparse
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# ===============================
# Add project root to sys.path
# so we can import the src package
# (src/__init__.py must exist).
# ===============================
CURRENT_DIR = os.path.dirname(__file__)                      # .../pipelines
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))  # project root

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Now we can import src.dataset_versions and its relative imports will work
from src.dataset_versions import load_dataset_version


def parse_args():
    """Parse command-line arguments passed to this Azure ML component."""
    parser = argparse.ArgumentParser(description="Stroke preprocessing component")

    parser.add_argument(
        "--raw_data",
        type=str,
        required=True,
        help="Path to the raw stroke dataset CSV (uri_file provided by Azure ML).",
    )

    parser.add_argument(
        "--version",
        type=str,
        required=False,
        default="v2_dropped",
        help="Dataset preprocessing version key (default: v2_dropped).",
    )

    # Three separate outputs: train / test / modified
    parser.add_argument(
        "--train_output",
        type=str,
        required=True,
        help="Output path for the train CSV (uri_file).",
    )

    parser.add_argument(
        "--test_output",
        type=str,
        required=True,
        help="Output path for the test CSV (uri_file).",
    )

    parser.add_argument(
        "--modified_output",
        type=str,
        required=True,
        help="Output path for the modified CSV (uri_file).",
    )

    return parser.parse_args()


def prepare_data(raw_df: pd.DataFrame, dataset_version: str):
    """
    Full end-to-end preprocessing with:
      - dataset version selection
      - feature engineering (inside load_dataset_version)
      - stratified train/valid/test split (70/15/15)
      - class weighting for imbalance

    Returns
    -------
    train_df : DataFrame
    valid_df : DataFrame
    test_df  : DataFrame
    class_weights : dict mapping class -> weight
    """
    # Apply version-specific preprocessing / feature engineering
    df = load_dataset_version(raw_df, dataset_version)

    # 70% train, 30% temp
    train_df, temp_df = train_test_split(
        df,
        stratify=df["stroke"],
        test_size=0.30,
        random_state=42,
    )

    # 15% valid, 15% test (split the remaining 30%)
    valid_df, test_df = train_test_split(
        temp_df,
        stratify=temp_df["stroke"],
        test_size=0.50,
        random_state=42,
    )

    # Compute class weights based on the training set
    counts = train_df["stroke"].value_counts().to_dict()
    total = sum(counts.values())
    n_classes = len(counts)
    class_weights = {cls: total / (n_classes * count) for cls, count in counts.items()}

    # Add a "weights" column to each split
    for df_part in [train_df, valid_df, test_df]:
        df_part["weights"] = df_part["stroke"].map(class_weights)

    return train_df, valid_df, test_df, class_weights


def main():
    args = parse_args()

    # 1. Read raw dataset
    input_path = args.raw_data
    print(f"[component] Reading raw data from: {input_path}")
    df_raw = pd.read_csv(input_path)
    print(f"[component] Raw shape: {df_raw.shape}")

    # 2. Run full preprocessing + stratified train/valid/test split
    print(f"[component] Applying dataset version with prepare_data: {args.version}")
    train_df, valid_df, test_df, class_weights = prepare_data(df_raw, args.version)

    print(f"[component] Train shape: {train_df.shape}")
    print(f"[component] Valid shape: {valid_df.shape}")
    print(f"[component] Test shape:  {test_df.shape}")
    print(f"[component] Class weights: {class_weights}")

    # 3. Create modified dataset based on the test set
    #    - age: +10 years
    #    - avg_glucose_level: +50% (multiply by 1.5)
    print("[component] Creating modified test dataset...")
    modified_df = test_df.copy()

    if "age" in modified_df.columns:
        modified_df["age"] = modified_df["age"] + 10

    if "avg_glucose_level" in modified_df.columns:
        modified_df["avg_glucose_level"] = (
            modified_df["avg_glucose_level"] * 1.5
        )

    print(f"[component] Modified shape: {modified_df.shape}")

    # 4. Write each dataset directly to the Azure ML uri_file outputs.
    #    For uri_file, Azure passes a full file path; we just write to it.
    train_path = args.train_output
    test_path = args.test_output
    modified_path = args.modified_output

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    modified_df.to_csv(modified_path, index=False)

    print(f"[component] Saved train data to: {train_path}")
    print(f"[component] Saved test data to: {test_path}")
    print(f"[component] Saved modified data to: {modified_path}")


if __name__ == "__main__":
    main()
