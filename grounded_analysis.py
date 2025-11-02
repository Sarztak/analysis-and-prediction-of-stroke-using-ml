from smoking_stroke_analysis import analyze_smoking_stroke_risk
import pandas as pd
import seaborn as sns
sns.set_theme(style="white")
stroke_data = pd.read_csv('healthcare-dataset-stroke-data.csv')
stroke_data.columns

stroke_data.sample(10)

stroke_data.drop(columns='id', inplace=True)

def age_stroke_dist(df):
    sns.histplot(data=df, x='age', hue='stroke')

age_stroke_dist(stroke_data)

"""## we bin the age into groups so that model can learn the non-linear increase in stroke cases with age"""

def age_transformation(df):
    df['age_group'] = pd.cut(
        df['age'],
        labels=["0-30", "30-60", "60+"],
        bins=[0, 30, 60, 100]
    )
    df['age_over_60'] = (df.age > 60).astype(int)
    df['age_over_80'] = (df.age > 80).astype(int)
    return df

stroke_data = age_transformation(stroke_data)
stroke_data

"""## Understand what the average glucose level even means in the first place and what medical significance does it have, where could the threshold be and validate it with data if such thresholding exists"""

def glucose_stroke_dist(df):
    sns.histplot(data=df, x='avg_glucose_level', hue='stroke')

glucose_stroke_dist(stroke_data)

"""## Let the data confirm whether domain logic applies here"""

def glucose_transformation(df):

    df['glucose_above_150'] = (df['avg_glucose_level'] > 150).astype(int)
    bins = [0, 100, 125, float('inf')]
    labels = ['normal', 'elevated', 'high']
    stroke_data['glucose_group'] = pd.cut(stroke_data['avg_glucose_level'], bins=bins, labels=labels)
    stroke_data['glucose_above_250'] = (stroke_data['avg_glucose_level'] > 250).astype(int)

    return df

stroke_data = glucose_transformation(stroke_data)

stroke_data

sns.histplot(data=stroke_data, x='bmi', hue='stroke')

stroke_data.isna().sum()

from bmi_viz import plot_bmi
plot_bmi(stroke_data)

from bmi_missingness_analysis import comprehensive_missingness_analysis
comprehensive_missingness_analysis(stroke_data)

import pandas as pd

def bmi_transform(df):
    # Define a function to categorize BMI
    def categorize_bmi(bmi):
        if bmi < 18.5:
            return 'Underweight'
        elif 18.5 <= bmi < 25.0:
            return 'Normal'
        elif 25.0 <= bmi < 30.0:
            return 'Overweight'
        elif 30.0 <= bmi < 35.0:
            return 'Obese (Class I)'
        elif 35.0 <= bmi < 40.0:
            return 'Obese (Class II)'
        else:
            return 'Morbid obesity'

    # Apply the categorization function to the 'bmi' column
    df['bmi_category'] = df['bmi'].apply(categorize_bmi)

    # Create the 'is_overweight' column
    df['is_overweight'] = (df['bmi'] > 25).astype(int)

    df["is_bmi_missing"] = df["bmi"].isnull().astype(int)

    return df

stroke_data = bmi_transform(stroke_data)
stroke_data

pd.pivot_table(data=stroke_data, index='smoking_status', columns='stroke', values='age', aggfunc='count')

stroke_data.groupby(['age_group', 'smoking_status'])['stroke'].count().unstack()

stroke_data.groupby(['age_group', 'smoking_status'])['stroke'].sum().unstack()

stroke_data.groupby(['age_group', 'smoking_status'])['stroke'].sum().unstack() / stroke_data.groupby(['age_group', 'smoking_status'])['stroke'].count().unstack()


analyze_smoking_stroke_risk(stroke_data)

"""## Unknown has a low stroke rate, but that low risk is due to age, not smoking,"""

def smoking_status_transformation(df):
    df['is_smoking_unknown'] = (df.smoking_status == 'Unknown').astype(int)
    df['is_smokes'] = (df.smoking_status == 'smokes').astype(int)
    return df
stroke_data = smoking_status_transformation(stroke_data)
stroke_data

"""Let’s apply the structured approach—this time, not to transform the target itself, but to understand how `imbalance` in the target can distort your `model’s learning` and `loss function`."""

stroke_data.value_counts(subset='stroke', dropna=False, normalize=True) * 100

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np

class BMICustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.imputer = IterativeImputer(random_state=42)

    def fit(self, X, y=None):
        bmi_data = X[["age", "bmi"]].to_numpy()
        self.imputer.fit(bmi_data)
        return self

    def transform(self, X):
        bmi_data = X[["age", "bmi"]].to_numpy()
        imputed = self.imputer.transform(bmi_data)
        X = X.copy()
        X["bmi"] = imputed[:, 1]
        return X

columns_to_use = [
    # 'gender',
    'age',

    # NOT USING FOR NOW
    # 'hypertension', 'heart_disease', 'ever_married',
    # 'work_type', 'Residence_type',

    'avg_glucose_level',
    'bmi',
    'smoking_status',
    'age_group',
    'age_over_60',
    'age_over_80',
    'glucose_above_150',
    'glucose_group',
    'glucose_above_250',
    'bmi_category',
    'is_overweight',
    'is_bmi_missing',
    'is_smoking_unknown',
    'is_smokes'
]

X = stroke_data[columns_to_use].copy()
y = stroke_data['stroke'].copy()

categorical_cols = X.select_dtypes(exclude=[int, float]).columns
col_to_impute = ['bmi']
categorical_cols

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, stratify=y
)

X_train.columns

X_train.value_counts('age_over_80')

pd.crosstab(X_train['age_over_60'], y_train, normalize='index').round(4)

pd.crosstab(X_train['age_over_80'], y_train, normalize='index').round(4)

stroke_data.value_counts('stroke', normalize=1)

sum(y_train), len(y_train)


pd.crosstab(X_train['glucose_above_250'], y_train)


pd.crosstab(X_train['glucose_above_150'], y_train)

pd.crosstab(X_train['is_bmi_missing'], y_train)

X_train.columns

X = stroke_data[[
    'age', 'avg_glucose_level', 'bmi',
       # 'age_over_60', 'glucose_above_150', 'is_bmi_missing'
       ]]
y = stroke_data['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y)

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.metrics import f1_score, classification_report, precision_recall_curve, auc, roc_auc_score, roc_curve

# age_order = ["0-30", "30-60", "60+"]
# glucose_order = ['normal', 'elevated', 'high']
# bmi_order = ['Underweight', 'Normal', 'Overweight', 'Obese (Class I)', 'Obese (Class II)', 'Morbid obesity']

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['gender', 'smoking_status']),
#         ('ordinal', OrdinalEncoder(
#             categories=[age_order, glucose_order, bmi_order]
#             ),
#             ['age_group', 'glucose_group', 'bmi_category']
#         ),
#     ],
#     remainder='passthrough'
# )


regularized_rf = RandomForestClassifier(
    n_estimators=100,          # Reasonable number of trees
    max_depth=5,               # Limit tree depth
    min_samples_split=3,      # Require more samples to split
    min_samples_leaf=5,       # Require more samples in leaf nodes
    # max_features='sqrt',       # Limit features considered per split
    random_state=98
)

pipeline_reg = Pipeline(steps=[
    ('rf_classifier', regularized_rf)
])

# Fit the regularized model
pipeline_reg.fit(X_train, y_train)

# Check training performance
y_train_pred = pipeline_reg.predict(X_train)
print("REGULARIZED MODEL - Training Performance:")
print(classification_report(y_train, y_train_pred))

# Check test performance
y_test_pred = pipeline_reg.predict(X_test)
print("\nREGULARIZED MODEL - Test Performance:")
print(classification_report(y_test, y_test_pred))