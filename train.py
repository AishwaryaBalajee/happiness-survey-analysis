import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import TRAIN_DATASET_PATH, input_cols, target_col
from logistic_regression import LogisticRegressionModel, LogisticRegressionFeatureSelectionModel

dataset = pd.read_csv(TRAIN_DATASET_PATH)

# Check if dataset is balanced -- Uncomment below
# print(dataset["Y"].value_counts())

# Input/Target
X = dataset[input_cols]  
y = dataset[target_col]

# Train Logistic Regression Model without feature selection
model_without_fs = LogisticRegressionModel()
model_without_fs.train(X, y)
# Save model
model_without_fs.save_model()

# Train Logistic Regression Model with feature selection
model_fs = LogisticRegressionFeatureSelectionModel()
model_fs.train(X, y)
model_fs.save_model()