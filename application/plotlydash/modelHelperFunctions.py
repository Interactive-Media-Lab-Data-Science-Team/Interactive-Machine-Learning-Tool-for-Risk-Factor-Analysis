import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, LassoLarsCV, ElasticNetCV, BayesianRidge, LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def regression_performance(y_true, y_pred):
    """Helper function to evaluate performance of regresson models
    Args:
        y_true (list): True label
        y_pred [list): Predicted label
    Return: MAE,MSE, R-squared
    """

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2


def classification_performance(y_true, y_pred):
    """Helper function to evaluate performance of classification models
    Args:
        y_true (list): True label
        y_pred (list): Predicted label
    Returns:
        Accuracy, ROC_AUC score, Precision, Recall, F1-Score
    """
    roc_auc = roc_auc_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return accuracy, roc_auc, precision, recall, f1


def regression_models(X, y, model_type, norm=False):
    """Regression models 
    Args:
        X (2D array): Regressor
        y (list): label
        model_type (String): "Linear", "Lasso", "Ridge", "LassoLars", "Bayesian Ridge", "Elastic Net"
        normalize (boolean): Normalize or not (if applicable) 
    Returns:
        regression model, MAE, MSE, R-squared, mean Gamma Deviance 
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    if model_type == "Linear":
        reg = LinearRegression(normalize=norm).fit(X_train, y_train)
    elif model_type == "Lasso":
        reg = LassoCV(cv=5, n_alphas=50, normalize=norm).fit(X_train, y_train)
    elif model_type == "Ridge":
        reg = RidgeCV(normalize=norm).fit(X_train, y_train)
    elif model_type == "LassoLars":
        reg = LassoLarsCV(normalize=norm).fit(X_train, y_train)
    elif model_type == "Bayesian Ridge":
        reg = BayesianRidge(normalize=norm).fit(X_train, y_train)
    elif model_type == "Elastic Net":
        reg = ElasticNetCV(normalize=norm).fit(X_train, y_train)
    else:
        return None
    y_pred = reg.predict(X_test)
    mae, mse, r2 = regression_performance(y_test, y_pred)
    return reg, mae, mse, r2


def classification_models(X, y, model_type, norm=False):
    """Classification models
    Args:
         X (2D array): Regressor
        y (list): label
        model_type (String): "Logistic", "LDA"
        normalize (boolean): Normalize or not (if applicable) 
    Returns:
        classification model, accuracy, roc_auc, precision, recall, f1
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    if norm:
        X_train = StandardScaler().fit_transform(X_train)
        X_test = StandardScaler().fit_transform(X_test)
    if model_type == "Logistic":
        clf = LogisticRegressionCV().fit(X_train, y_train)
    elif model_type == "LDA":
        clf = LinearDiscriminantAnalysis().fit(X_train, y_train)
    else:
        return None
    y_pred = clf.predict(X_test)
    accuracy, roc_auc, precision, recall, f1 = classification_performance(
        y_test, y_pred)

    return clf, accuracy, roc_auc, precision, recall, f1


def risk_factor_analysis(model, cols):
    """Return Risk Factor Analysis table
        Args:
            model (model object): risk factor analysis model
            cols (list): feature name list
        Returns:
            list of dict: Risk Factor Analysis table
        """
    coef = model.coef_.reshape(-1)
    sign = np.where(np.array(coef) > 0, '+', '-').tolist()
    sort_coef = sorted(np.abs(coef), reverse=True)
    sort_index = sorted(
        range(len(coef)), key=lambda k: np.abs(coef)[k], reverse=True)
    return [{"Rank": i+1, "Factor": cols[sort_index[i]], "ABS weight": round(sort_coef[i], 5), "sign": sign[sort_index[i]]} for i in range(len(coef))]
