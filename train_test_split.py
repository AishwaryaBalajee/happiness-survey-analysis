import pandas as pd
from sklearn.model_selection import train_test_split
from config import DATASET_PATH, TRAIN_DATASET_PATH, TEST_DATASET_PATH

dataset = pd.read_csv(DATASET_PATH)
dataset['X5*X6'] = (dataset['X5'] * dataset['X6']).astype(int)

# Input/Target
X = dataset[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X5*X6']]  
y = dataset['Y']

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
