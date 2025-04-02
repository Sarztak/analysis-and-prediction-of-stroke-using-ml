import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")


schema = {
    'id': pl.Int64,
    'gender': pl.Categorical,
    'age': pl.Float64,
    'hypertension': pl.Int64,
    'heart_disease': pl.Int64,
    'ever_married': pl.Categorical,
    'work_type': pl.Categorical,
    'Residence_type': pl.Categorical,
    'avg_glucose_level': pl.Float64,
    'bmi': pl.Float64,
    'smoking_status': pl.Categorical,
    'stroke': pl.Categorical
}


df = pl.read_csv(
    'healthcare-dataset-stroke-data.csv',
    null_values=['Unknown', 'N/A'],
    schema=schema,
    columns=list(range(1, 12))
)


df.describe()

# the age of children is quoted as float, not int, which creates problems

sns.histplot(data=df, x='bmi')

sns.countplot(data=df, x='gender')

df.select(pl.col('gender').value_counts())


# there is only one other value; I can impute it to the
# most common one that is female

# check distribution of age; what section of population is represented here; but before that we have to change the age to integer values
# change the name of Resident_type column

df = df.rename({'Residence_type': 'residence_type'})

target = ['stroke'] 

categorical_cols = [
    'gender',
    'hypertension',
    'heart_disease',
    'ever_married',
    'work_type',
    'residence_type',
    'smoking_status',
]


# do not use multiplot during data exploration because setting the correct configuration takes too much time, use single plots instead
N = len(categorical_cols)
ncols = 2
nrows = N // ncols if N % ncols == 0 else N // ncols + 1 
fig = plt.figure(figsize=(15, 13))
fig.subplots_adjust(hspace=0.8, wspace=0.3)

for idx, col in enumerate(categorical_cols): 
    fig.add_subplot(nrows, ncols, idx + 1)
    sns.countplot(data=df, x=col)


# plot the target: the data is imbalanced and that is expected. Not many people 
sns.countplot(data=df, x='stroke')

# the total number of datapoints
len(df)

df.columns

# this is a bimodal distribution
sns.histplot(df, x='avg_glucose_level')

# age needs to be converted to integer first or not. Should I keep it as it is ? 
# those who are children should they be part of this dataset ? What is the chance that they will get stroke ? On the other hand older people are at risk for stroke 
sns.histplot(df, x='age', binwidth=1)

# looks like in this dataset we have slightly more people between 40 to 60 age group. How does the age group affect the cases of stroke ? 
sns.kdeplot(df, x='age', hue='stroke', alpha=0.7)

# clearly age is a great predictor of stroke, people aged 40 and above account for almost all the cases of stroke. This is expected. Body degrades. But there may be some abnormal cases masked as well.

sns.histplot(df.filter(pl.col('stroke') == 1), x='age')

# there are a few cases where stroke had been reported for age 2, 15 perhaps

df.filter((pl.col('stroke') == 1) & (pl.col('age') <= 30)).select('age')

# are these outliers ? may be not; one thing to understand is that this dataset may contain cases that are not due to natural causes. May be due to accidents or during surgery etc. So be careful while interpreting the results

# how many cases do we have of less than 2 year ? There was a spike in the histogram
df.filter(pl.col('age') < 3).select('age').count()

# 175 cases.

# what is the distribution ? 
sns.histplot(
    df.filter(pl.col('age') < 3),
    x='age'
)

# if I were to truncate all the cases less than 1 to 1 and those less than 2 and between 1 will it cause damage ? may be not, the data is too large. Do we need prediction for ages which are not integers ? may be, should I keep it as it is. Perhaps I should. It does not harm

# bmi vs stroke
sns.kdeplot(
    df,
    x='bmi',
    hue='stroke'
) # those who had stroke had average bmi that is between 20 to 40

# avg_glucose_type
sns.kdeplot(
    df,
    x='avg_glucose_level',
    hue='stroke'
)

g = sns.FacetGrid(df, col='stroke', hue='stroke')
g.map(sns.kdeplot, 'bmi')

g = sns.FacetGrid(df, col='stroke', hue='stroke')
g.map(sns.kdeplot, 'avg_glucose_level')


g = sns.FacetGrid(df, col='stroke', hue='stroke')
g.map(sns.kdeplot, 'age')



























































































































