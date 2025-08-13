import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import DATASET_PATH, input_cols, target_col
from logistic_regression import logistic_regression, logistic_regression_with_p_value_and_feature_selection

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


# Scale data and convert back to pd dataframe
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=input_cols)
X_test = pd.DataFrame(scaler.transform(X_test), columns=input_cols)

# reset indices for target data to match the indices of input data (drop the old indices)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


model_without_fs = logistic_regression(X_train, X_test, y_train, y_test)
model_with_fs = logistic_regression_with_p_value_and_feature_selection(X_train, X_test, y_train, y_test)
# logistic_regression_with_p_value_and_feature_selection(X, X_test, y, y_test)