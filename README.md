This is a readme file, and now I am making changed to this file as I speak and committing it to master branch. Amen
This is a readme file. Now I am making changes to the readme file in the feature branch and committing it to feature branch. This should trigger merge conflict when I try to merge feature into the master branch

Now for the sake of rebase I am making changes and committing to the master branch
This is a readme file. Now I am making changes to the readme file in the feature branch and committing it to feature branch. This should trigger merge conflict when I try to merge feature into the master branch

Now I am making changes to the feature branch readme for rebasing. this should trigger a conflict that will be resolved by rebase

### Data-cleaning file organized correctly

| Section | Function             | Purpose                               |
| ------- | -------------------- | ------------------------------------- |
| **0**   | `handle_outliers`    | Numeric sanity clipping and flagging  |
| **1**   | `BMICustomImputer`   | BMI imputation based on age           |
| **2**   | `clean_categoricals` | Category consolidation, flag creation |
| **3**   | `drop_missing_rows`  | Hard drop of NA records               |
| **4**   | `remove_identifier`  | Remove non-predictive IDs             |
| **5**   | `full_cleaning`      | Orchestrates all the above            |


### Feature Creation organized correctly
| Section | Function             | Purpose                               |
| ------- | -------------------- | ------------------------------------- |
| 1       | `create_age_features()`     | Adds age bands and binary flags for older groups            |
| 2       | `create_glucose_features()` | Adds medically-motivated glucose groups and threshold flags |
| 3       | `create_bmi_features()`     | Adds BMI category, missingness, and overweight indicators   |
| 4       | `create_smoking_features()` | Creates binary indicator for current smokers                |


### What various files do ?
| Layer | File             | Purpose                               |
| ------- | -------------------- | ------------------------------------- |
| **Data Cleaning**                | Validates and fixes data            | `data_cleaning.py`    |
| **Feature Creation**             | Builds domain features              | `feature_creation.py` |
| **Transformation Orchestration** | Combines both for configurable runs | `transformation.py`   |
| **Model Preprocessing**          | Scaling, encoding, modeling         | `preprocessing.py`    |
