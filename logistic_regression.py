from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import numpy as np
from config import pval_threshold

def logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression()
    model.fit(X_train,y_train)
    
    accuracy = model.score(X_test, y_test)
    print("Accuracy on Test Set without Feature Selection:", accuracy)

    return model
    
    
def logistic_regression_with_p_value_and_feature_selection(X_train, X_test, y_train, y_test):
    # model = LogisticRegression(X,y)
    # Add intercept
    X_train_sm = sm.add_constant(X_train)
    logit_model = sm.Logit(y_train, X_train_sm)
    result = logit_model.fit()
    pvals = result.pvalues.drop('const')
    significant_pvals = pvals[pvals<pval_threshold].index.tolist()
    X_train, X_test = X_train[significant_pvals], X_test[significant_pvals]
    
    model = LogisticRegression()
    model.fit(X_train,y_train)
    
    accuracy = model.score(X_test, y_test)
    print("Selected Features:", significant_pvals)
    print("Accuracy on Test Set with Feature Selection:", accuracy)

    return model, significant_pvals