import pandas as pd
import numpy as np

# -------------------------------------------------------------------------
# 1. Age-related Features
# -------------------------------------------------------------------------
def create_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates age groups and binary flags for older populations."""
    df = df.copy()

    df["age_group"] = pd.cut(
        df["age"], bins=[0, 30, 60, 100], labels=["0-30", "30-60", "60+"]
    ).astype("category")
    df["age_over_60"] = (df["age"] > 60).astype(int)
    df["age_over_80"] = (df["age"] > 80).astype(int)
    return df


# -------------------------------------------------------------------------
# 2. Glucose-related Features
# -------------------------------------------------------------------------
def create_glucose_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds categorical and binary glucose indicators."""
    df = df.copy()

    bins = [0, 100, 125, np.inf]
    labels = ["normal", "elevated", "high"]
    df["glucose_group"] = pd.cut(df["avg_glucose_level"], bins=bins, labels=labels).astype("category")

    df["glucose_above_150"] = (df["avg_glucose_level"] > 150).astype(int)
    df["glucose_above_250"] = (df["avg_glucose_level"] > 250).astype(int)
    return df


# -------------------------------------------------------------------------
# 3. BMI-related Features
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

    df["bmi_category"] = df["bmi"].apply(categorize_bmi).astype("category")
    df["is_overweight"] = (df["bmi"] > 25).astype(int)

    return df


# -------------------------------------------------------------------------
# 4. Smoking-related Features
# -------------------------------------------------------------------------
def create_smoking_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates binary indicators from smoking status."""
    df = df.copy()
    df["is_smokes"] = (df["smoking_status"] == "smokes").astype(int)
    return df

# -------------------------------------------------------------------------
# 5. Master Feature Creation Function
# -------------------------------------------------------------------------
def apply_all_feature_creation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all domain-specific feature creation functions in sequence.
    Produces a fully engineered feature set ready for preprocessing.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset (after data_cleaning.full_cleaning()).

    Returns
    -------
    df : pd.DataFrame
        DataFrame with all engineered features added.
    """
    df = df.copy()

    df = create_age_features(df)
    df = create_glucose_features(df)
    df = create_bmi_features(df)
    df = create_smoking_features(df)

    return df


if __name__ == "__main__":
    from data_cleaning import full_cleaning
    raw_df = pd.read_csv("./data/stroke_data.csv")
    df_clean = full_cleaning(raw_df)
    df_features = apply_all_feature_creation(df_clean)

