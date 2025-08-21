import pandas as pd
from config import TRAIN_DATASET_PATH, input_cols, target_col

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from logistic_regression import LogisticRegressionModel, LogisticRegressionFeatureSelectionModel, LogisticRegressionFeatureSelectionStackedModel


dataset = pd.read_csv(TRAIN_DATASET_PATH)

X = dataset[input_cols]  
y = dataset[target_col]

# rf = RandomForestClassifier(n_estimators=200, random_state=42)
# xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# rf_score = cross_val_score(rf, X, y, cv=5, scoring='accuracy').mean()
# xgb_score = cross_val_score(xgb, X, y, cv=5, scoring='accuracy').mean()

# print("RF CV Accuracy:", rf_score)
# print("XGBoost CV Accuracy:", xgb_score)

model_stacked = LogisticRegressionFeatureSelectionStackedModel()
model_stacked.get_rf_important_features(X, y)
