"""
Assembles the full transformation workflow by combining
data cleaning and feature creation stages.
"""

import pandas as pd
from data_cleaning import full_cleaning
from feature_creation import apply_all_feature_creation


def assemble_feature_set(
    df: pd.DataFrame,
    apply_cleaning: bool = True,
    impute_bmi: bool = True,
    clip_outliers: bool = True,
    drop_missing: bool = False,
) -> pd.DataFrame:
    """
    Orchestrates the complete transformation pipeline:
    cleaning (optional) → feature creation.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataset.
    apply_cleaning : bool, default=True
        Whether to perform cleaning before feature creation.
    impute_bmi : bool, default=True
        Whether to impute BMI during cleaning.
    clip_outliers : bool, default=True
        Whether to cap extreme BMI and glucose values.
    drop_missing : bool, default=False
        Whether to drop all rows with missing values.

    Returns
    -------
    df : pd.DataFrame
        Fully cleaned and feature-engineered dataset.
    """
    df = df.copy()

    # Step 1: cleaning
    if apply_cleaning:
        df = full_cleaning(df, impute_bmi=impute_bmi, clip_outliers=clip_outliers, drop_missing=drop_missing)

    # Step 2: feature creation
    df = apply_all_feature_creation(df)

    return df


if __name__ == "__main__":
    raw_df = pd.read_csv("./data/stroke_data.csv")

    # Example versions
    v0_raw = assemble_feature_set(raw_df, apply_cleaning=False)
    v1_clipped = assemble_feature_set(raw_df, impute_bmi=False, clip_outliers=True)
    v2_dropped = assemble_feature_set(raw_df, impute_bmi=False, clip_outliers=True, drop_missing=True)
    v3_imputed = assemble_feature_set(raw_df, impute_bmi=True, clip_outliers=True)
