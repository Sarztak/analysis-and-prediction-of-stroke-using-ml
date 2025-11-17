import sys
import os

# file is run from the parent directory py model/baseline.py
sys.path.append(os.getcwd())

from features.transformation import assemble_feature_set

import pandas as pd 
import h2o; h2o.init()
from h2o.automl import H2OAutoML
from rich.traceback import install; install()


import h2o
from h2o.automl import H2OAutoML
from sklearn.model_selection import train_test_split
import pandas as pd

def train_linear_model(raw_df):
    # 1. Feature assembly
    df_featurized = assemble_feature_set(raw_df, drop_missing=True)

    # 2. Stratified train–test split in pandas/sklearn (H2O lacks native stratify)
    train_df, test_df = train_test_split(
        df_featurized, stratify=df_featurized["stroke"], test_size=0.2, random_state=42
    )

    # 3. Compute class weights = inverse of class frequency
    counts = train_df["stroke"].value_counts().to_dict()
    total = sum(counts.values())
    n_classes = len(counts)
    class_weights = {cls: total / (n_classes * count) for cls, count in counts.items()}

    # Map weights to rows
    train_df["weights"] = train_df["stroke"].map(class_weights)

    # 4. Convert to H2OFrames
    h2o.init()
    h2o_train = h2o.H2OFrame(train_df)
    h2o_test = h2o.H2OFrame(test_df)

    # ensure factor for classification
    h2o_train['stroke'] = h2o_train['stroke'].asfactor()
    h2o_test['stroke'] = h2o_test['stroke'].asfactor()

    y = "stroke"
    x = [col for col in h2o_train.columns if col not in [y, "weights"]]

    # 5. AutoML
    aml = H2OAutoML(
        max_models=50,
        seed=1984,
        stopping_metric="AUCPR",  # aligns with PR-based objective
        sort_metric="AUCPR",
        balance_classes=False, # stratification used 
        max_runtime_secs=900,
    )

    # Train with weights (reweighting, no resampling)
    aml.train(x=x, y=y, training_frame=h2o_train, weights_column="weights")

    # Evaluate on test set
    aml.leaderboard.head()
    leader = aml.leader
    test_perf = leader.model_performance(h2o_test)
    print("Test AUCPR:", test_perf.aucpr())

    target_precision = 0.90
    threshold = test_perf.find_threshold_by_max_metric("precision")
    max_precision = test_perf.metric("precision", [threshold])[0][1]
    print(f"Threshold @{max_precision}% precision:{threshold}")
    print(f"Recall at this threshold:{test_perf.recall(threshold)[0][1]:.3f}",)


    return aml, h2o_train, h2o_test

if __name__ == "__main__":
    raw_df = pd.read_csv("./data/stroke_data.csv")
    aml, X_train, X_test = train_linear_model(raw_df)
    lb_all = aml.leaderboard # leaderboard contains model results
    lb_all = lb_all.as_data_frame(use_pandas=True) # convert to pandas df for easy wrangling
    lb_all.head() # all metrics are computed on X_val internally


