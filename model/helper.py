import numpy as np
import sys
import os
sys.path.append(os.getcwd())
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import dotenv

env_dict = dotenv.dotenv_values('./.env')
random_state = int(env_dict.get('RANDOM_STATE'))
train_ratio = float(env_dict.get('TRAIN_RATIO'))
val_ratio = float(env_dict.get('VAL_RATIO'))
test_ratio = float(env_dict.get('TEST_RATIO'))

def compute_train_test_val_split(df):
    
    X = df.drop(columns=["stroke"])
    y = df["stroke"]
    
    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=(val_ratio + test_ratio), 
        stratify=y, 
        random_state=random_state
    )
    
    # Second split: val vs test from the temp set
    # test_ratio / (val_ratio + test_ratio) gives the proportion of test in temp
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=test_ratio / (val_ratio + test_ratio), 
        stratify=y_temp, 
        random_state=random_state,
    )
    
    splits = dict(
        train=(X_train, y_train),
        val=(X_val, y_val),
        test=(X_test, y_test),
    )
    
    return splits

def compute_class_weights(y_train):
    counts = y_train.value_counts().to_dict()
    total = sum(counts.values())
    n_classes = len(counts)
    class_weights = {_cls: total / (n_classes * n_count) for _cls, n_count in counts.items()}
    sample_weights = y_train.map(class_weights).values
    return sample_weights

if __name__ == "__main__":
    raw_df = pd.read_csv("./data/stroke_data.csv")
    splits = compute_train_test_val_split(raw_df)
    X_train, y_train = splits.get('train')
    sample_weights = compute_class_weights(y_train)