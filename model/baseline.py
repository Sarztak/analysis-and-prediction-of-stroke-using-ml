import sys
import os

# file is run from the parent directory py model/baseline.py
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from rich.traceback import install
from features.transformation import assemble_feature_set
from features.preprocessing import make_training_pipeline
from sklearn.calibration import CalibratedClassifierCV
import time 

install()


def train_linear_model(raw_df):

    # can be used for Logistic Regression Baseline
    df_featurized = assemble_feature_set(
        raw_df, drop_missing=True
    )

    X = df_featurized.drop(columns=["stroke"])
    y = df_featurized["stroke"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=1971
    )

    reg_model = LogisticRegression(class_weight='balanced')
    calibrated_model = CalibratedClassifierCV(
        reg_model,
        method='isotonic', # what does having large samples mean
        cv=5
    )

    pipeline = make_training_pipeline(calibrated_model, X_train, drop_first=True)
    pipeline.fit(X_train, y_train)

    # 5. Make predictions    
    y_train_prob = pipeline.predict_proba(X_train)
    y_test_prob = pipeline.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= 0.1).astype(int)
    
    # # 6. Calculate metrics
    # metrics = {
    #     'train': {
    #         'f1_score': f1_score(y_train, y_train_pred),
    #         'precision': precision_score(y_train, y_train_pred),
    #         'recall': recall_score(y_train, y_train_pred),
    #     },
        
    #     'test': {
    #         'f1_score': f1_score(y_test, y_test_pred),
    #         'precision': precision_score(y_test, y_test_pred),
    #         'recall': recall_score(y_test, y_test_pred),
    #     },
    # }

    print(classification_report(y_test, y_test_pred))

    # return model, metrics, prep_info, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    raw_df = pd.read_csv("./data/stroke_data.csv")
    train_linear_model(raw_df)