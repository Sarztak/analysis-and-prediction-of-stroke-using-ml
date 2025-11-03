import pandas as pd
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
# 3. Drop Missing Values
# -------------------------------------------------------------------------
def drop_missing_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop all rows containing missing values"""
    return df.dropna().reset_index(drop=True)


# -------------------------------------------------------------------------
# 4. Remove Identifier Variables
# -------------------------------------------------------------------------
def remove_identifier(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier columns that should not be used for modeling."""
    df = df.copy()
    for col in ["id", "patient_id", "record_id"]:
        if col in df.columns:
            df = df.drop(columns=col)
    return df

# -------------------------------------------------------------------------
# 4. Cleaning Function
# -------------------------------------------------------------------------

def full_cleaning(
        df: pd.DataFrame, 
        impute_bmi=True, 
        clip_outliers=True, 
        drop_missing=False
    ) -> pd.DataFrame:
    """
    Cleaning up data with optional stages.

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
        Processed DataFrame ready for feature engineering or storage.
    """
    df = df.copy()
    df = remove_identifier(df)
    df = clean_categoricals(df)
    if clip_outliers:
        df = handle_outliers(df)
    if drop_missing:
        df = drop_missing_rows(df)
    elif impute_bmi:
        df = BMICustomImputer().fit_transform(df)
    return df


if __name__ == "__main__":
    raw_df = pd.read_csv("./data/stroke_data.csv")

    # v0_raw
    v0_raw = full_cleaning(raw_df, impute_bmi=False, clip_outliers=False, drop_missing=False)

    # v1_clipped
    v1_clipped = full_cleaning(raw_df, impute_bmi=False, clip_outliers=True, drop_missing=False)

    # v2_dropped
    v2_dropped = full_cleaning(raw_df, impute_bmi=False, clip_outliers=True, drop_missing=True)

    # v3_imputed_full
    v3_imputed_full = full_cleaning(raw_df, impute_bmi=True, clip_outliers=True, drop_missing=False)
