import sys
import os
import mlflow
import argparse

# file is run from the parent directory py model/baseline.py
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, average_precision_score, roc_auc_score
from rich.traceback import install; install()
from features.transformation import assemble_feature_set
from features.preprocessing import make_training_pipeline
from sklearn.calibration import CalibratedClassifierCV

def train_linear_model(raw_df, run_name, experiment_name="baseline_logistic_regression", random_state=1984):

    mlflow.set_experiment(experiment_name=experiment_name) # set name of the experiment

    # can be used for Logistic Regression Baseline
    df_featurized = assemble_feature_set(
        raw_df, drop_missing=True
    )

    X = df_featurized.drop(columns=["stroke"])
    y = df_featurized["stroke"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    reg_model = LogisticRegression(random_state=random_state) # I don't need class weights for the baseline model

    with mlflow.start_run(run_name=run_name):

        mlflow.log_param("random_state", random_state)
        mlflow.log_param("class_weights", False)
        pipeline = make_training_pipeline(reg_model, X_train, drop_first=True)
        pipeline.fit(X_train, y_train)

        # 5. Make predictions    
        y_test_prob = pipeline.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_prob >= 0.5).astype(int)
        
        # 6. Calculate metrics
        metrics = {
            'f1_score': f1_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred),
            'aucpr': average_precision_score(y_test, y_test_prob),
            'roc_auc': roc_auc_score(y_test, y_test_prob)
        }

        mlflow.log_metrics(metrics) 


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run a baseline experiment using Logistic Regression with no class weights")
    parser.add_argument('-r', '--run_name', help="The name of the run")
    args = parser.parse_args()
    run_name = args.run_name

    raw_df = pd.read_csv("./data/stroke_data.csv")

    # run the command in a separate terminal to start the server
    # mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5050

    # set tracking uri
    mlflow.set_tracking_uri("http://127.0.0.1:5050")

    # train the model
    train_linear_model(raw_df=raw_df, run_name=run_name)
