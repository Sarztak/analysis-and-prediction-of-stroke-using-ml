# src/features/transformations.py
"""
Feature engineering transformations for the Stroke Risk project.
Includes data cleaning, domain-specific feature creation, and encoding helpers.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer


# -------------------------------------------------------------------------
# 0. Outlier Handling (applied before feature engineering)
# -------------------------------------------------------------------------

def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Caps extreme numeric outliers at medically plausible limits
    and retains binary flags for analysis/modeling.
    """
    df = df.copy()

    # --- BMI ---
    # Flag unrealistic BMI (<10 or >60)
    df["bmi_outlier_flag"] = (df["bmi"] < 10) | (df["bmi"] > 60)
    # Clip within safe physiological range
    df["bmi"] = df["bmi"].clip(lower=10, upper=60)

    # --- Average Glucose Level ---
    # Flag potential errors (<40 or >300)
    df["glucose_outlier_flag"] = (df["avg_glucose_level"] < 40) | (df["avg_glucose_level"] > 300)
    # Clip to clinically reasonable limits
    df["avg_glucose_level"] = df["avg_glucose_level"].clip(lower=40, upper=300)

    return df

# -------------------------------------------------------------------------
# 1. Custom BMI Imputer
# -------------------------------------------------------------------------
class BMICustomImputer(BaseEstimator, TransformerMixin):
    """Imputes BMI using Iterative Imputer based on Age."""

    def __init__(self, random_state=42):
        self.imputer = IterativeImputer(random_state=random_state)

    def fit(self, X, y=None):
        bmi_data = X[["age", "bmi"]].to_numpy()
        self.imputer.fit(bmi_data)
        return self

    def transform(self, X):
        X = X.copy()
        bmi_data = X[["age", "bmi"]].to_numpy()
        imputed = self.imputer.transform(bmi_data)
        X["bmi"] = imputed[:, 1]
        return X


# -------------------------------------------------------------------------
# 2. Cleaning Categorical Variables
# -------------------------------------------------------------------------
def clean_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans rare and inconsistent categories."""

    df = df.copy()

    # Gender: merge 'Other' → 'Female'
    df["gender"] = df["gender"].replace("Other", "Female")

    # Work Type: collapse rare categories
    df["work_type"] = df["work_type"].replace(
        {"children": "Other", "Never_worked": "Other"}
    )

    # Smoking: keep as-is but create Unknown flag
    df["is_smoking_unknown"] = (df["smoking_status"] == "Unknown").astype(int)

    return df


# -------------------------------------------------------------------------
# 3. Age-related Features
# -------------------------------------------------------------------------
def create_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates age groups and binary flags for older populations."""
    df = df.copy()

    df["age_group"] = pd.cut(
        df["age"], bins=[0, 30, 60, 100], labels=["0–30", "30–60", "60+"]
    )
    df["age_over_60"] = (df["age"] > 60).astype(int)
    df["age_over_80"] = (df["age"] > 80).astype(int)
    return df


# -------------------------------------------------------------------------
# 4. Glucose-related Features
# -------------------------------------------------------------------------
def create_glucose_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds categorical and binary glucose indicators."""
    df = df.copy()

    bins = [0, 100, 125, np.inf]
    labels = ["normal", "elevated", "high"]
    df["glucose_group"] = pd.cut(df["avg_glucose_level"], bins=bins, labels=labels)

    df["glucose_above_150"] = (df["avg_glucose_level"] > 150).astype(int)
    df["glucose_above_250"] = (df["avg_glucose_level"] > 250).astype(int)
    return df


# -------------------------------------------------------------------------
# 5. BMI-related Features
# -------------------------------------------------------------------------
def create_bmi_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds BMI category, missingness flags, and overweight indicator."""
    df = df.copy()

    # Flag missingness first
    df["is_bmi_missing"] = df["bmi"].isna().astype(int)

    # Categorize BMI
    def categorize_bmi(bmi):
        if pd.isna(bmi):
            return np.nan
        if bmi < 18.5:
            return "Underweight"
        elif 18.5 <= bmi < 25:
            return "Normal"
        elif 25 <= bmi < 30:
            return "Overweight"
        elif 30 <= bmi < 35:
            return "Obese I"
        elif 35 <= bmi < 40:
            return "Obese II"
        else:
            return "Morbid Obesity"

    df["bmi_category"] = df["bmi"].apply(categorize_bmi)
    df["is_overweight"] = (df["bmi"] > 25).astype(int)

    return df


# -------------------------------------------------------------------------
# 6. Smoking-related Features
# -------------------------------------------------------------------------
def create_smoking_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates binary indicators from smoking status."""
    df = df.copy()
    df["is_smokes"] = (df["smoking_status"] == "smokes").astype(int)
    return df


# -------------------------------------------------------------------------
# 7. Master Transformation Function
# -------------------------------------------------------------------------
def assemble_feature_set(
    df: pd.DataFrame,
    impute_bmi: bool = True,
    clip_outliers: bool = True,
    drop_missing: bool = False,
) -> pd.DataFrame:
    """
    Runs the complete feature engineering pipeline with optional stages.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataset.
    impute_bmi : bool, default=True
        Whether to impute BMI values using BMICustomImputer.
        Set False to keep BMI missing values as-is (for raw or dropped versions).
    clip_outliers : bool, default=True
        Whether to cap extreme BMI and glucose values and add outlier flags.
    drop_missing : bool, default=False
        Whether to drop all rows with missing values instead of imputing.

    Returns
    -------
    df : pd.DataFrame
        Processed DataFrame ready for modeling or storage.
    """

    df = df.copy()

    # Drop ID column if present
    if "id" in df.columns:
        df = df.drop(columns="id")

    # Step 1: Handle numeric outliers
    if clip_outliers:
        df = handle_outliers(df)

    # Step 2: Drop missing rows (alternative to imputation)
    if drop_missing:
        df = df.dropna().reset_index(drop=True)

    # Step 3: Clean categorical inconsistencies
    df = clean_categoricals(df)

    # Step 4: BMI imputation (only if not dropping missing)
    if impute_bmi and not drop_missing:
        imputer = BMICustomImputer()
        df = imputer.fit_transform(df)

    # Step 5: Feature creation
    df = create_age_features(df)
    df = create_glucose_features(df)
    df = create_bmi_features(df)
    df = create_smoking_features(df)

    return df


# -------------------------------------------------------------------------
# 8. Utility — Column Grouping (optional for later encoding)
# -------------------------------------------------------------------------
def get_feature_groups(df: pd.DataFrame):
    """Returns feature type groupings for preprocessing."""
    numeric_features = ["age", "avg_glucose_level", "bmi"]
    categorical_features = [
        "gender",
        "work_type",
        "residence_type",
        "ever_married",
        "smoking_status",
        "age_group",
        "glucose_group",
        "bmi_category",
    ]
    binary_features = [
        "hypertension",
        "heart_disease",
        "age_over_60",
        "age_over_80",
        "glucose_above_150",
        "glucose_above_250",
        "is_overweight",
        "is_bmi_missing",
        "is_smoking_unknown",
        "is_smokes",
    ]
    return numeric_features, categorical_features, binary_features


if __name__ == "__main__":
    raw_df = pd.read_csv("./data/stroke_data.csv")

    # v0_raw
    v0_raw = assemble_feature_set(raw_df, impute_bmi=False, clip_outliers=False, drop_missing=False)

    # v1_clipped
    v1_clipped = assemble_feature_set(raw_df, impute_bmi=False, clip_outliers=True, drop_missing=False)

    # v2_dropped
    v2_dropped = assemble_feature_set(raw_df, impute_bmi=False, clip_outliers=True, drop_missing=True)

    # v3_imputed_full
    v3_imputed_full = assemble_feature_set(raw_df, impute_bmi=True, clip_outliers=True, drop_missing=False)
