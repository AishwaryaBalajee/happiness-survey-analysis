import pandas as pd
from config import TRAIN_DATASET_PATH, input_cols, target_col
from logistic_regression import LogisticRegressionModel, LogisticRegressionFeatureSelectionModel, LogisticRegressionFeatureSelectionStackedModel

dataset = pd.read_csv(TRAIN_DATASET_PATH)

# Check if dataset is balanced -- Uncomment below
# print(dataset["Y"].value_counts())

# Input/Target
X = dataset[input_cols]  
y = dataset[target_col]

# Train Logistic Regression Model without feature selection
print('-----------Logistic Regression------------------')
model_without_fs = LogisticRegressionModel()
model_without_fs.train(X, y)
# Save model
model_without_fs.save_model()

# Train Logistic Regression Model with feature selection
print('-----------Logistic Regression Feature Selection------------------')
model_fs = LogisticRegressionFeatureSelectionModel()
model_fs.train(X, y)
model_fs.save_model()

# Train Logistic Regression Model with feature selection
print('-----------Logistic Regression Model Stacked Model with feature selection------------------')
model_stacked = LogisticRegressionFeatureSelectionStackedModel()
model_stacked.train(X, y)
model_stacked.save_model()