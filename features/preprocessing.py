import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from transformations import get_feature_groups


# -------------------------------------------------------------------------
# 1. ColumnTransformer Builder
# -------------------------------------------------------------------------
def build_preprocessor(df: pd.DataFrame):
    """
    Creates a preprocessing pipeline with scaling, encoding, and passthroughs.
    Returns a ColumnTransformer ready to be plugged into a model pipeline.
    """

    numeric_features, categorical_features, binary_features = get_feature_groups(df)

    # Pipelines for each type
    numeric_transformer = Pipeline(
        steps=[
            # ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            # ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=None),
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
def make_training_pipeline(model, df: pd.DataFrame):
    """
    Combines preprocessing with the given model into a full pipeline.
    model: any scikit-learn estimator (e.g., RandomForestClassifier)
    Returns: sklearn.Pipeline
    """

    preprocessor = build_preprocessor(df)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from transformations import assemble_feature_set
    from sklearn.model_selection import train_test_split
    from rich.traceback import install 

    install()

    df = pd.read_csv("./data/stroke_data.csv")
    df = assemble_feature_set(df)
    X = df.drop(columns=["stroke"])
    y = df["stroke"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1984)
    pipeline = make_training_pipeline(rf_model, X_train)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
