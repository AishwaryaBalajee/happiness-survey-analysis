from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import joblib
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import pval_threshold, input_cols, target_col, save_model_dir

# def logistic_regression(X_train, X_test, y_train, y_test):
#     model = LogisticRegression()
#     model.fit(X_train,y_train)
    
#     accuracy = model.score(X_test, y_test)
#     print("Accuracy on Test Set without Feature Selection:", accuracy)

#     return model

# def logistic_regression_with_p_value_and_feature_selection(X_train, X_test, y_train, y_test):
#     # Add intercept
#     X_train_sm = sm.add_constant(X_train)
#     logit_model = sm.Logit(y_train, X_train_sm)
#     result = logit_model.fit()
#     pvals = result.pvalues.drop('const')
#     significant_pvals = pvals[pvals<pval_threshold].index.tolist()
#     X_train, X_test = X_train[significant_pvals], X_test[significant_pvals]
    
#     model = LogisticRegression()
#     model.fit(X_train,y_train)
    
#     accuracy = model.score(X_test, y_test)
#     print("Selected Features:", significant_pvals)
#     print("Accuracy on Test Set with Feature Selection:", accuracy)

#     return model, significant_pvals

class LogisticRegressionModel():
    def __init__(self, *args, **kwargs):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(**kwargs)
        self.is_trained = False
        self.columns = None
    
    def train(self, X_train, y_train):
        self.columns = X_train.columns.tolist()
        X_train = pd.DataFrame(self.scaler.fit_transform(X_train), columns=self.columns)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print('Training Complete')
    
    def save_model(self, save_model_dir=save_model_dir):
        path = os.path.join(save_model_dir, 'logistic_regression.pkl')
        joblib.dump({"scaler": self.scaler, "columns": self.columns, "model": self.model}, path)
        print('Model Saved at: ', path)
    
    def test(self, X_test, y_test):
        if self.is_trained == False:
            raise RuntimeError("Model must be trained before testing.")
        else:
            X_test = pd.DataFrame(self.scaler.transform(X_test[self.columns]), columns=input_cols)
            accuracy = self.model.score(X_test, y_test)
            print("Accuracy on Test Set without Feature Selection:", accuracy)
            return accuracy
    def saved_model_test(self, X_test, y_test, model_path):
        saved_objects = joblib.load(model_path)
        saved_scaler, saved_columns, saved_model = saved_objects['scaler'], saved_objects["columns"], saved_objects['model']
        X_test = pd.DataFrame(saved_scaler.transform(X_test[saved_columns]), columns=saved_columns)
        accuracy = saved_model.score(X_test, y_test)
        print("Accuracy on Test Set without Feature Selection:", accuracy)
        return accuracy
    
class LogisticRegressionFeatureSelectionModel():
    def __init__(self, *args, **kwargs):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(**kwargs)
        self.is_trained = False
        self.columns = None
        self.features = None
    
    def get_coefficients_pvalues(self, X_train, y_train):
        self.columns = X_train.columns.tolist()
        X_train = pd.DataFrame(self.scaler.fit_transform(X_train), columns=self.columns)
        X_train_sm = sm.add_constant(X_train)
        logit_model = sm.Logit(y_train, X_train_sm)
        result = logit_model.fit()
        pvalues = result.pvalues.drop('const')
        return X_train, y_train, result.params, pvalues
    
    def feature_selection(self, X_train, y_train):
        X_train, y_train, coefficients, pvalues = self.get_coefficients_pvalues(X_train, y_train)
        self.features = pvalues[pvalues<pval_threshold].index.tolist()
    
    def train(self, X_train, y_train):
        self.feature_selection(X_train, y_train)
        X_train = pd.DataFrame(self.scaler.fit_transform(X_train[self.features]), columns=self.features)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print('Training Complete')
    
    def save_model(self, save_model_dir=save_model_dir):
        path = os.path.join(save_model_dir, 'logistic_regression_feature_selection.pkl')
        joblib.dump({"scaler": self.scaler, "columns": self.features, "model": self.model}, path)
        print('Model Saved at: ', path)
    
    def test(self, X_test, y_test):
        if self.is_trained == False:
            raise RuntimeError("Model must be trained before testing.")
        else:
            X_test = pd.DataFrame(self.scaler.transform(X_test[self.features]), columns=self.features)
            accuracy = self.model.score(X_test, y_test)
            print("Accuracy on Test Set with Feature Selection:", accuracy)
            return accuracy
    
    def saved_model_test(self, X_test, y_test, model_path):
        saved_objects = joblib.load(model_path)
        saved_scaler, saved_columns, saved_model = saved_objects['scaler'], saved_objects["columns"], saved_objects['model']
        X_test = pd.DataFrame(saved_scaler.transform(X_test[saved_columns]), columns=saved_columns)
        accuracy = saved_model.score(X_test, y_test)
        print("Accuracy on Test Set with Feature Selection:", accuracy)
        return accuracy
    
        
        
