from config import DATASET_PATH
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegressionFeatureSelectionModel

# Dataset
dataset = pd.read_csv(DATASET_PATH)
df = dataset.copy()
df['X5*X6'] = (df['X5'] * df['X6']).astype(int)

X = df[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X5*X6']]  
y = df['Y']

# Check for missing values
print('Null Values in dataset')
print(df.isnull().sum())
print('\n')

# Check the value ranges
print(df.describe())
print('\n')

# Check if dataset is balanced
print('Y value Count')
print(df['Y'].value_counts(normalize=True))
print('\n')

# Univariate Analysis
fig, axes = plt.subplots(3,3, figsize = (15,8))
axes = axes.flatten()

for i,col in enumerate(X.columns):
    sns.countplot(data=df, x=col, ax=axes[i])
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_xlabel("Response (1-5)")
    axes[i].set_ylabel("Count")
plt.suptitle("Univariate Analysis of Survey Features", fontsize=16)
plt.tight_layout()
plt.show(block=False)

plt.figure(figsize=(5,4))
sns.countplot(data=df, x='Y', palette="Set2")
plt.title("Distribution of Target (Happy vs Unhappy)")
plt.xlabel("Y (0 = unhappy, 1 = happy)")
plt.ylabel("Count")
plt.show(block=False)

# Bivariate Analysis
fig, axes = plt.subplots(3,3, figsize = (15,8))
axes = axes.flatten()

for i,col in enumerate(X.columns):
    sns.boxplot(data=df, x='Y', y=col, ax=axes[i])
    axes[i].set_title(f"{col} vs Happiness")

plt.suptitle("Biivariate Analysis of Survey Features", fontsize=16)
plt.tight_layout()
plt.show(block=False)

# Mean responses by target
print('Mean Responses by Target')
print(df.groupby('Y')[['X1','X2','X3','X4','X5','X6', 'X5*X6']].mean())

# Correlation Matrix
corr = df.corr()
plt.figure(figsize=(5,4))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title("Feature Correlations with Happiness")
plt.show(block=False)

# Multivariate Analysis
sns.pairplot(df, hue='Y')
plt.show()

# model_fs = LogisticRegressionFeatureSelectionModel()
# pvalues, coefficients = model_fs.feature_selection(X, y)
# print('coefficients:')
# print(coefficients)
# print('P values:')
# print(pvalues)


# This loop is to keep the code running for the plots to stay open
while plt.get_fignums():
    plt.pause(0.5)