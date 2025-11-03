import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

sns.set_theme(style="whitegrid")


def load_data(filepath: str) -> pl.DataFrame:
    schema = {
        'id': pl.Int64,
        'gender': pl.Categorical,
        'age': pl.Float64,
        'hypertension': pl.Int64,
        'heart_disease': pl.Int64,
        'ever_married': pl.Categorical,
        'work_type': pl.Categorical,
        'residence_type': pl.Categorical,
        'avg_glucose_level': pl.Float64,
        'bmi': pl.Float64,
        'smoking_status': pl.Categorical,
        'stroke': pl.Categorical
    }
    
    df = pl.read_csv(
        filepath,
        null_values=['Unknown', 'N/A'],
        schema=schema,
        columns=list(range(1, 12))
    )
    
    # the age of children is quoted as float, not int, which creates problems
    return df.rename({'residence_type': 'residence_type'})


def explore_categorical_features(df: pl.DataFrame, categorical_cols: list) -> None:
    # do not use multiplot during data exploration because setting the correct configuration takes too much time, use single plots instead
    N = len(categorical_cols)
    ncols = 2
    nrows = N // ncols if N % ncols == 0 else N // ncols + 1 
    fig = plt.figure(figsize=(15, 13))
    fig.subplots_adjust(hspace=0.8, wspace=0.3)
    
    for idx, col in enumerate(categorical_cols): 
        fig.add_subplot(nrows, ncols, idx + 1)
        sns.countplot(data=df, x=col)


def explore_age_distribution(df: pl.DataFrame) -> None:
    # age needs to be converted to integer first or not. Should I keep it as it is ? 
    # those who are children should they be part of this dataset ? What is the chance that they will get stroke ? On the other hand older people are at risk for stroke 
    sns.histplot(df, x='age', binwidth=1)
    
    # looks like in this dataset we have slightly more people between 40 to 60 age group. How does the age group affect the cases of stroke ? 
    sns.kdeplot(df, x='age', hue='stroke', alpha=0.7)
    
    # clearly age is a great predictor of stroke, people aged 40 and above account for almost all the cases of stroke. This is expected. Body degrades. But there may be some abnormal cases masked as well.
    sns.histplot(df.filter(pl.col('stroke') == '1'), x='age')


def analyze_young_stroke_cases(df: pl.DataFrame) -> None:
    # there are a few cases where stroke had been reported for age 2, 15 perhaps
    df.filter((pl.col('stroke') == '1') & (pl.col('age') <= 30)).select('age')
    
    # are these outliers ? may be not; one thing to understand is that this dataset may contain cases that are not due to natural causes. May be due to accidents or during surgery etc. So be careful while interpreting the results
    
    # how many cases do we have of less than 2 year ? There was a spike in the histogram
    df.filter(pl.col('age') < 3).select('age').count()

def analyze_bmi_missingness(df: pl.DataFrame) -> None:
    # convert to pandas dataframe for missingno compatibility
    pdf = df.to_pandas()

    # Standard missingness analysis
    plt.figure(figsize=(10, 6))
    msno.matrix(pdf)
    plt.title('Missing Values Heatmap')
    plt.show()
    
    # Create a DataFrame with the missing indicator
    plot_df = df.with_columns(
        pl.col('bmi').is_null().alias('bmi_missing')
    ).to_pandas()

    # Age vs BMI missingness analysis
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df.filter(pl.col('bmi').is_null()),
        x='age',
        binwidth=5
    )
    plt.title('Distribution of Age for Missing BMI Values')
    plt.show()

    # Compare age distributions
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=plot_df,
        x='bmi_missing',
        y='age'
    )
    plt.title('Age Distribution: Missing vs Non-missing BMI')
    plt.xlabel('BMI is missing')
    plt.show()

    # Print summary statistics
    print("\nAge statistics for missing BMI:")
    print(df.filter(pl.col('bmi').is_null()).select('age').describe())
    print("\nAge statistics for non-missing BMI:")
    print(df.filter(pl.col('bmi').is_not_null()).select('age').describe())

    # Correlation of missingness between variables
    plt.figure(figsize=(10, 6))
    msno.heatmap(pdf)  # Changed from df to pdf
    plt.title('Correlation of Missingness')
    plt.show()

    # Nullity bar chart
    plt.figure(figsize=(10, 6))
    msno.bar(pdf)  # Changed from df to pdf
    plt.title('Nullity by Column')
    plt.show()

def impute_missing_bmi(df: pl.DataFrame) -> pl.DataFrame:
    # after going down this rabbit hole, I am back
    # the main question was how do I impute the values of bmi, there are only 201 missing. But before that we need to think what caused it. 
    
    missing_bmi = df.filter(pl.col('bmi').is_null())
    
    # age seems to be one of the causes of missingness; it could be difficult to collect the data for old people or may be it does not matter for them; this shows that missingness is not random, so I cannot throw away the data. I can, but I will have to justify why I did so.
    
    bmi_imputation = df[["age","bmi"]].to_numpy()
    imputer = IterativeImputer(random_state=42)
    imputed_data = imputer.fit_transform(bmi_imputation)
    
    return df.with_columns(pl.Series('bmi_imputed', imputed_data[:, 1]))


def main():
    # Load and prepare initial data
    df = load_data('healthcare-dataset-stroke-data.csv')
    
    # categorical_cols = [
    #     'gender', 'hypertension', 'heart_disease', 'ever_married',
    #     'work_type', 'residence_type', 'smoking_status',
    # ]
    # numerical_cols = ['age', 'avg_glucose_level', 'bmi']
    # target = ['stroke']
    
    # df.describe()
    # sns.histplot(data=df, x='bmi')
    # sns.countplot(data=df, x='gender')
    # df.select(pl.col('gender').value_counts())
    
    # explore_categorical_features(df, categorical_cols)
    # explore_age_distribution(df)
    # analyze_young_stroke_cases(df)
    
    # analyze missingness of bmi column; imputation depends on the missingness pattern
    analyze_bmi_missingness(df)
    
    # # Handle missing values
    # df_imputed = impute_missing_bmi(df) KekeKe I smell data leakage !!!
    
    # X_train, X_test, y_train, y_test = train_test_split(
    #     df.select(categorical_cols + numerical_cols),
    #     df.select(target),
    #     test_size=0.2,
    #     random_state=1984
    # )
    
    # cv = KFold(n_splits=5, shuffle=True, random_state=1984)
    # dummy_clf = DummyClassifier(strategy='uniform')
    # dummy_scores = cross_val_score(dummy_clf, X_train, y_train, cv=cv, scoring='f2')

if __name__ == "__main__":
    main()








































































