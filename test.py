import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import TEST_DATASET_PATH, input_cols, target_col, save_model_dir
from logistic_regression import LogisticRegressionModel, LogisticRegressionFeatureSelectionModel

dataset = pd.read_csv(TEST_DATASET_PATH)

# Check if dataset is balanced -- Uncomment below
# print(dataset["Y"].value_counts())

# Input/Target
X_test = dataset[input_cols]  
y_test = dataset[target_col]

# Test Logistic Regression Model without feature selection
model_without_fs = LogisticRegressionModel()
model_path = os.path.join(save_model_dir, 'logistic_regression.pkl')
model_without_fs.saved_model_test(X_test, y_test, model_path)

# Test Logistic Regression Model with feature selection
model_with_fs = LogisticRegressionFeatureSelectionModel()
model_path = os.path.join(save_model_dir, 'logistic_regression_feature_selection.pkl')
model_with_fs.saved_model_test(X_test, y_test, model_path)