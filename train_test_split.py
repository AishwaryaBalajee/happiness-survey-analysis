import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import input_cols, target_col, DATASET_PATH, TRAIN_DATASET_PATH, TEST_DATASET_PATH
from logistic_regression import LogisticRegressionModel, LogisticRegressionFeatureSelectionModel

dataset = pd.read_csv(DATASET_PATH)

# Check if dataset is balanced -- Uncomment below
# print(dataset["Y"].value_counts())

# Input/Target
X = dataset[input_cols]  
y = dataset[target_col]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.15,      # 15% for testing
    random_state=42,    # reproducibility
    stratify=y          # keep class balance same in train & test
)

# Reset Indices
X_train, X_test = X_train.reset_index(drop = True), X_test.reset_index(drop = True)
y_train, y_test = y_train.reset_index(drop = True), y_test.reset_index(drop = True)

# Combine X and y for train and test sets
train_df = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
test_df = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)

# Save to CSV
train_df.to_csv(TRAIN_DATASET_PATH, index=False)
test_df.to_csv(TEST_DATASET_PATH, index=False)
