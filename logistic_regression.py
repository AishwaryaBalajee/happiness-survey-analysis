from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import statsmodels.api as sm
import joblib
import os
import numpy as np
import pandas as pd
from config import pval_threshold, save_model_dir,save_results_dir, rf_importance_threshold, xgb_importance_threshold, lr_coefficient_threshold

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
        X_train_scaled = pd.DataFrame(self.scaler.fit_transform(X_train), columns=self.columns)
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        print('Training Complete')
    
    def save_model(self, save_model_dir=save_model_dir):
        path = os.path.join(save_model_dir, 'logistic_regression.pkl')
        joblib.dump({"scaler": self.scaler, "columns": self.columns, "model": self.model}, path)
        print('Model Saved at: ', path)
    
    def test(self, X_test, y_test, save_results):
        if self.is_trained == False:
            raise RuntimeError("Model must be trained before testing.")
        else:
            X_test_scaled = pd.DataFrame(self.scaler.transform(X_test[self.columns]), columns=self.columns)
            accuracy = self.model.score(X_test_scaled, y_test)
            print("Accuracy on Test Set without Feature Selection:", accuracy)
            
            if save_results == True:
                y_pred = self.model.predict(X_test_scaled)
                results_df = X_test.copy()
                results_df['y_true'] = y_test
                results_df['y_pred'] = y_pred
                results_df.to_csv(os.path.join(save_results_dir, 'test_with_feature_selection.csv'), index=False)
            return accuracy
    def saved_model_test(self, X_test, y_test, model_path, save_results = True):
        saved_objects = joblib.load(model_path)
        saved_scaler, saved_columns, saved_model = saved_objects['scaler'], saved_objects["columns"], saved_objects['model']
        X_test_scaled = pd.DataFrame(saved_scaler.transform(X_test[saved_columns]), columns=saved_columns)
        accuracy = saved_model.score(X_test_scaled, y_test)
        print("Accuracy on Test Set without Feature Selection:", accuracy)
        
        # Save predictions
        if save_results == True:
            y_pred = saved_model.predict(X_test_scaled)
            results_df = X_test.copy()
            results_df['y_true'] = y_test
            results_df['y_pred'] = y_pred
            results_df.to_csv(os.path.join(save_results_dir, 'test_without_feature_selection.csv'), index=False)
        
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
        X_train_scaled = pd.DataFrame(self.scaler.fit_transform(X_train), columns=self.columns)
        X_train_sm = sm.add_constant(X_train_scaled)
        logit_model = sm.Logit(y_train, X_train_sm)
        result = logit_model.fit()
        # result = logit_model.fit_regularized(method='l1', alpha=0.1)
        pvalues = result.pvalues.drop('const')
        return result.params, pvalues
    
    def feature_selection(self, X_train, y_train):
        coefficients, pvalues = self.get_coefficients_pvalues(X_train, y_train)
        self.features = coefficients[abs(coefficients)>lr_coefficient_threshold].index.tolist()
        print('Features Selected are: ', self.features)
        return pvalues, coefficients
    
    def train(self, X_train, y_train):
        pvalues, coefficients = self.feature_selection(X_train, y_train)
        X_train_scaled = pd.DataFrame(self.scaler.fit_transform(X_train[self.features]), columns=self.features)
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        print('Training Complete')
    
    def save_model(self, save_model_dir=save_model_dir):
        path = os.path.join(save_model_dir, 'logistic_regression_feature_selection.pkl')
        joblib.dump({"scaler": self.scaler, "columns": self.features, "model": self.model}, path)
        print('Model Saved at: ', path)
    
    def test(self, X_test, y_test, save_results = True):
        if self.is_trained == False:
            raise RuntimeError("Model must be trained before testing.")
        else:
            X_test_scaled = pd.DataFrame(self.scaler.transform(X_test[self.features]), columns=self.features)
            accuracy = self.model.score(X_test_scaled, y_test)
            print("Accuracy on Test Set with Feature Selection:", accuracy)
            
            if save_results == True:
                y_pred = self.model.predict(X_test_scaled)
                results_df = X_test.copy()
                results_df['y_true'] = y_test
                results_df['y_pred'] = y_pred
                results_df.to_csv(os.path.join(save_results_dir, 'test_with_feature_selection.csv'), index=False)
            return accuracy
    
    def saved_model_test(self, X_test, y_test, model_path, save_results = True):
        saved_objects = joblib.load(model_path)
        saved_scaler, saved_columns, saved_model = saved_objects['scaler'], saved_objects["columns"], saved_objects['model']
        X_test_scaled = pd.DataFrame(saved_scaler.transform(X_test[saved_columns]), columns=saved_columns)
        accuracy = saved_model.score(X_test_scaled, y_test)
        print('Features selected are: ', saved_columns)
        print("Accuracy on Test Set with Feature Selection:", accuracy)
        
        # Save predictions
        if save_results == True:
            y_pred = saved_model.predict(X_test_scaled)
            results_df = X_test.copy()
            results_df['y_true'] = y_test
            results_df['y_pred'] = y_pred
            results_df.to_csv(os.path.join(save_results_dir, 'test_with_feature_selection.csv'), index=False)
        return accuracy
    

class LogisticRegressionFeatureSelectionStackedModel():
    def __init__(self, *args, **kwargs):
        self.base_lr = LogisticRegression(**kwargs)
        self.rf = RandomForestClassifier(n_estimators=500, random_state=42)
        self.xgb = XGBClassifier(
            # n_estimators=500,
            # learning_rate=0.05,
            # max_depth=5,
            # subsample=0.8,
            # colsample_bytree=0.8,
            # random_state=42,
            # use_label_encoder=False,
            # eval_metric="logloss"
        )
        self.is_trained = False
        self.stacked_model = None
        self.columns = None
        self.lr_features = None
        self.rf_features = None
        self.xgb_features = None
    
    def get_lr_coefficients_pvalues(self, X_train, y_train):
        scaler = StandardScaler()
        self.columns = X_train.columns.tolist()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=self.columns)
        X_train_sm = sm.add_constant(X_train_scaled)
        logit_model = sm.Logit(y_train, X_train_sm)
        result = logit_model.fit()
        # result = logit_model.fit_regularized(method='l1', alpha=0.1)
        pvalues = result.pvalues.drop('const')
        return result.params, pvalues      
        
    def feature_selection(self, X_train, y_train):
        coefficients, pvalues = self.get_lr_coefficients_pvalues(X_train, y_train)
        self.lr_features = coefficients[abs(coefficients)>lr_coefficient_threshold].index.tolist()
        print('LR feature Coefficients:\n ', coefficients)
        print('Features Selected for LR: ', self.lr_features)
        self.rf.fit(X_train, y_train)
        rf_importances = pd.Series(self.rf.feature_importances_, index=X_train.columns)
        self.rf_features = rf_importances[rf_importances>rf_importance_threshold].index.tolist()
        print('RF feature importances:\n ', rf_importances)
        print('Features Selected for RF: ', self.rf_features)
        self.xgb.fit(X_train, y_train)
        xgb_importances = pd.Series(self.xgb.feature_importances_, index=X_train.columns)
        self.xgb_features = xgb_importances[xgb_importances>xgb_importance_threshold].index.tolist()
        print('XGB feature importances:\n ', xgb_importances)
        print('Features Selected for XGB: ', self.xgb_features)
    
    def train(self, X_train, y_train):
        self.feature_selection(X_train, y_train)
        
        lr_pipeline = Pipeline([('select_lr_features', ColumnTransformer([('lr_features', 'passthrough', self.lr_features)], remainder='drop')), 
                                ('scaler', StandardScaler()), ('lr', self.base_lr)])
        rf_pipeline = Pipeline([('select_rf_features', ColumnTransformer([('rf_features', 'passthrough', self.rf_features)], remainder='drop')), 
                                ('rf', self.rf)])
        xgb_pipeline = Pipeline([('select_xgb_features', ColumnTransformer([('xgb_features', 'passthrough', self.xgb_features)], remainder='drop')), 
                                ('xgb', self.xgb)])
        self.stacked_model = StackingClassifier(
            estimators=[
                ('lr', lr_pipeline),
                ('rf' ,rf_pipeline),
                ('xgb' ,xgb_pipeline)
            ],
            final_estimator=LogisticRegression(),
            cv=5,
            # stack_method='predict'
        )
        
        self.stacked_model.fit(X_train, y_train)
        self.is_trained = True
        print('Training Complete')
    
    def save_model(self, save_model_dir=save_model_dir):
        path = os.path.join(save_model_dir, 'logistic_regression_stacked.pkl')
        joblib.dump({'lr_features': self.lr_features, 'rf_features': self.rf_features, 'xgb_features': self.xgb_features, "model": self.stacked_model}, path)
        print('Model Saved at: ', path)
    
    def test(self, X_test, y_test, save_results = True):
        if self.is_trained == False:
            raise RuntimeError("Model must be trained before testing.")
        else:
            accuracy = self.stacked_model.score(X_test, y_test)
            print("Accuracy on Test Set with Feature Selection:", accuracy)
            
            if save_results == True:
                y_pred = self.stacked_model.predict(X_test)
                results_df = X_test.copy()
                results_df['y_true'] = y_test
                results_df['y_pred'] = y_pred
                results_df.to_csv(os.path.join(save_results_dir, 'test_with_feature_selection.csv'), index=False)
            return accuracy
    
    def saved_model_test(self, X_test, y_test, model_path, save_results = True):
        saved_objects = joblib.load(model_path)
        saved_lr_features, saved_rf_features, saved_xgb_features, saved_model = saved_objects['lr_features'], saved_objects['rf_features'], saved_objects['xgb_features'], saved_objects['model']
        accuracy = saved_model.score(X_test, y_test)
        
        print('Features selected for LR are: ', saved_lr_features)
        print('Features selected for RF are: ', saved_rf_features)
        print('Features selected for XGB are: ', saved_xgb_features)
        print("Accuracy on Test Set with Stacked Mode;:", accuracy)
        
        lr_pred = saved_model.named_estimators_['lr'].predict(X_test)
        rf_pred = saved_model.named_estimators_['rf'].predict(X_test)
        xgb_pred = saved_model.named_estimators_['xgb'].predict(X_test)
        print("LR vs RF predictions correlation:", np.corrcoef(lr_pred, rf_pred)[0,1])
        print("RF vs XGB predictions correlation:", np.corrcoef(rf_pred, xgb_pred)[0,1])
        print("LR vs XGB predictions correlation:", np.corrcoef(lr_pred, xgb_pred)[0,1])
        print('Model Output Coefficients [LR, RF, XGB]: ', saved_model.final_estimator_.coef_)
        
        # Save predictions
        if save_results == True:
            y_pred = saved_model.predict(X_test)
            results_df = X_test.copy()
            results_df['y_true'] = y_test
            results_df['y_pred'] = y_pred
            results_df.to_csv(os.path.join(save_results_dir, 'test_with_stacked_model.csv'), index=False)
        return accuracy
