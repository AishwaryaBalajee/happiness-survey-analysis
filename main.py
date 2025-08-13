import pandas as pd
from sklearn.model_selection import train_test_split
from config import DATASET_PATH

dataset = pd.read_csv(DATASET_PATH)

# Check if dataset is balanced -- Uncomment below
# print(dataset["Y"].value_counts())

X = dataset[["X1", "X2", "X3", "X4", "X5", "X6"]]  
y = dataset["Y"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for testing
    random_state=42,    # reproducibility
    stratify=y          # keep class balance same in train & test
)

# Target