from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def get_model(model_name: str):
    model_name = model_name.lower()

    if model_name == "xgb":
        return XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=20,
            eval_metric="logloss",
        )

    elif model_name == "lightgbm":
        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            class_weight="balanced"
        )

    elif model_name == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            class_weight="balanced"
        )

    elif model_name == "logreg":
        return LogisticRegression(max_iter=3000, class_weight="balanced")

    elif model_name == "svm":
        return SVC(kernel="rbf", class_weight="balanced", probability=True)

    else:
        raise ValueError(f"Unknown model name: {model_name}")
