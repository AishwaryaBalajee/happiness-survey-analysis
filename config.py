DATASET_PATH = r'C:\Users\aishu\Desktop\Apziva_projects\FL5MQSmgo61RrqZ5\dataset\ACME-HappinessSurvey2020.csv'

# Where to save and retrive train & test data
TRAIN_DATASET_PATH = r"C:\Users\aishu\Desktop\Apziva_projects\FL5MQSmgo61RrqZ5\dataset\train.csv"
TEST_DATASET_PATH = r"C:\Users\aishu\Desktop\Apziva_projects\FL5MQSmgo61RrqZ5\dataset\test.csv"

input_cols =["X1", "X2", "X3", "X4", "X5", "X6"]
target_col = "Y"

# Threshold for logistic regression
pval_threshold = 0.1
rf_importance_threshold = 0.16
xgb_importance_threshold = 0.18

# Where to save models and results
save_model_dir = r'C:\Users\aishu\Desktop\Apziva_projects\FL5MQSmgo61RrqZ5\models'
save_results_dir = r'C:\Users\aishu\Desktop\Apziva_projects\FL5MQSmgo61RrqZ5\results'