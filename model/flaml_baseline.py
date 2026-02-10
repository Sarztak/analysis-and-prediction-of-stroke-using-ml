from rich.traceback import install; install()
import argparse
import sys
import os
sys.path.append(os.getcwd())

from helper import compute_class_weights, compute_train_test_val_split
from features.transformation import assemble_feature_set
import pandas as pd 
import mlflow
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, average_precision_score
import dotenv


# file is run from the parent directory py model/baseline.py
env_dict = dotenv.dotenv_values("./.env")
RANDOM_STATE = int(env_dict.get('RANDOM_STATE'))
TIME_BUDGET = int(env_dict.get('TIME_BUDGET'))
estimators = env_dict.get('estimators')

def train_automl_model(raw_df, run_name):
    # 1. Feature assembly
    df_featurized = assemble_feature_set(raw_df, drop_missing=True)

    splits = compute_train_test_val_split(df_featurized)
    X_train, y_train = splits.get('train')
    X_val, y_val = splits.get('val')
    X_test, y_test = splits.get('test')

    sample_weights = compute_class_weights(y_train)
    

    # 4. Set Experiment 
    mlflow.set_experiment("flaml_baseline")

    # 5. Start the run
    with mlflow.start_run(run_name=run_name):
        automl = AutoML()

        estimator_list = estimators.split(',')

        automl.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            # sample_weight=sample_weights,
            metric='ap',
            task='classification',
            estimator_list=estimator_list,
            time_budget=TIME_BUDGET,
            seed=RANDOM_STATE,
        )

        # Log parent run parameters
        parent_run_params = dict(
            time_budget=TIME_BUDGET,
            metric='ap',
            estimators=estimators,
            seed=RANDOM_STATE,
            best_estimator=automl.best_estimator,
        )

        # log best config
        mlflow.log_dict(automl.best_config, "best_config.json")

        # evaluation of validation and test set
        y_pred_proba_val = automl.predict_proba(X_val)[:, 1]
        val_metrics = dict(
          val_aucpr=average_precision_score(y_val, y_pred_proba_val),
          val_roc_auc=roc_auc_score(y_val, y_pred_proba_val)
        )
        
        y_pred_proba_test = automl.predict_proba(X_test)[:, 1]
        test_metrics = dict(
          test_aucpr=average_precision_score(y_test, y_pred_proba_test),
          test_roc_auc=roc_auc_score(y_test, y_pred_proba_test)
        )

        mlflow.log_metrics(val_metrics)        
        mlflow.log_metrics(test_metrics)        
   

if __name__ == "__main__":
    # set tracking uri
    mlflow.set_tracking_uri("http://127.0.0.1:5050")

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run_name', help='name of the experiment to run')
    args = parser.parse_args()
    
    raw_df = pd.read_csv("./data/stroke_data.csv")
    train_automl_model(raw_df, run_name=args.run_name)