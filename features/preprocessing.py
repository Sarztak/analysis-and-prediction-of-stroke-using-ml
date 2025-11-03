import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# -------------------------------------------------------------------------
# 0. Utility — Column Grouping
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


# -------------------------------------------------------------------------
# 1. ColumnTransformer Builder
# -------------------------------------------------------------------------
def build_preprocessor(df: pd.DataFrame, scale_numeric=True, drop_first=False):
    """
    Creates a preprocessing pipeline with scaling, encoding, and passthroughs.
    Returns a ColumnTransformer ready to be plugged into a model pipeline.
    scale_numeric=False → tree ensembles, gradient boosting.
    scale_numeric=True → linear, logistic, SVM, MLP, etc.
    drop_first=True -> linear, logistic to avoid multi-collinear variables
    drop_first=False -> tree ensembles, gradient boosting.
    """

    numeric_features, categorical_features, binary_features = get_feature_groups(df)
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append((("scaler", StandardScaler())))
     
    # Pipelines for each type
    numeric_transformer = Pipeline(numeric_steps)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop= 'first' if drop_first else None),
            ),
        ]
    )

    # Binary features are already 0/1, passthrough
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
            ("bin", "passthrough", binary_features),
        ]
    )

    return preprocessor


# -------------------------------------------------------------------------
# 2. Pipeline Assembler
# -------------------------------------------------------------------------
def make_training_pipeline(model, df: pd.DataFrame, scale_numeric=True, drop_first=False):
    """
    Combines preprocessing with the given model into a full pipeline.
    model: any scikit-learn estimator (e.g., RandomForestClassifier)
    Returns: sklearn.Pipeline
    """

    preprocessor = build_preprocessor(df, scale_numeric=True, drop_first=drop_first)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from data_cleaning import full_cleaning
    from feature_creation import apply_all_feature_creation
    from rich.traceback import install 
    import matplotlib.pyplot as plt 

    install()

    df_raw = pd.read_csv("./data/stroke_data.csv")
    df_cleaned = full_cleaning(df_raw)
    df_featurized = apply_all_feature_creation(df_cleaned)

    X = df_featurized.drop(columns=["stroke"])
    y = df_featurized["stroke"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1984)
    pipeline = make_training_pipeline(rf_model, X_train, drop_first=True)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    feature_names = pipeline.named_steps['preprocess'].get_feature_names_out()
    feature_imp = pipeline.named_steps['model'].feature_importances_
    plt.barh(feature_names, feature_imp)
