import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from data_cleaning import full_cleaning
from feature_creation import apply_all_feature_creation
from preprocessing import make_training_pipeline
from model_factory import get_model
import argparse

def train_model(model_name: str):

    print(f"=== Training model: {model_name} ===")

    # ---- Load + Clean ----
    df_raw = pd.read_csv("./data/stroke_data.csv")
    df_cleaned = full_cleaning(df_raw)
    df_featurized = apply_all_feature_creation(df_cleaned)

    # ---- Split ----
    X = df_featurized.drop(columns=["stroke"])
    y = df_featurized["stroke"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ---- Model ----
    model = get_model(model_name)

    # ---- Pipeline ----
    pipeline = make_training_pipeline(model, X_train, drop_first=True)

    # ---- Train ----
    pipeline.fit(X_train, y_train)

    # ---- Predict ----
    y_pred = pipeline.predict(X_test)

    # ---- Report ----
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    train_model(args.model)