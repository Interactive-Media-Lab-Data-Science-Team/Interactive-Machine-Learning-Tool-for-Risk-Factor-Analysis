import numpy as np
import pandas as pd 
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, precision_score, f1_score, log_loss
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score
# supervised model (classifier)
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#import xgboost as xgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def model_performance(y_true, y_pred):
    """Helper function: Calculate common classification model performance criterions. Support multi_class
    Args:
        y_true (list): True label
        y_pred (list): Predicted label
    Returns:
        Accuracy, ROC_AUC score, Precision, F1-Score, Confusion Matrix
    """
    roc_auc = roc_auc_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return [accuracy, roc_auc, precision, f1, cm]





def logistic_regression(X, y, C, penalty, solver):
    """Logistic regression with cross validation. Assume input data is normalized
    Args:
        X : input data
        y : label 
        C ([int]): inverse of regularization strength
        penalty ([String]): penalty method: {'l1', 'l2', 'elasticnet'}
        solver ([String]): Solver for optimization {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}
                    
    Returns:
        [list]: Model performance listed:  Accuracy, ROC_AUC score, Precision, F1-Score, Confusion Matrix, cross-entropy
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = LogisticRegressionCV(Cs=C, penalty=penalty,solver=solver).fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    ce_loss = log_loss(y_test,y_prob)
    res = model_performance(y_test, y_pred)
    res += [ce_loss]
    res += ["Logistic Regression with C = {}, penalty = {}, solver = {}".format(C,penalty,solver)]

    return res

def random_forest(X, y, n_est, max_dep, min_sample, max_features, criterion):
    """Random forest
    Args:
        X : input data
        y : label 
        n_est ([int]): number of trees
        max_dep ([int]): max depth of trees
        min_sample ([int]): minimum number of samples at leaf
        max_features ([Srting]): number of features to be best split {“auto”, “sqrt”, “log2”}
        criterion ([String]): {“gini”, “entropy”}
    Returns:
        [list]: Model performance listed:  Accuracy, ROC_AUC score, Precision, F1-Score, Confusion Matrix, cross-entropy
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = RandomForestClassifier(n_estimators=n_est, max_depth=max_dep, min_samples_leaf=min_sample,\
         max_features= max_features, criterion=criterion,random_state=0)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    ce_loss = log_loss(y_test,y_prob)
    res = model_performance(y_test, y_pred)
    res += [ce_loss]
    return res

# def xgboost(X, y, lr, n_est, subsample, max_features, max_depth):
#     param_dist = {'objective':'binary:logistic', 'n_estimators':2}
# evaluation_GBC(X,y,parameter_gbc,model_name)


def gbc(X, y, loss, learning_rate,n_est,subsample, criterion,max_depth):
    """[Gradient Boosting]
    Args:
        X : input data
        y : target
        loss : loss function type, deviance or exponential {‘deviance’, ‘exponential’}
        learning_rate (folat): shrinks the contribution of each tree. defualt=0.1 
        n_est (int): The number of boosting stages to perform, default = 100
        subsample (float): The fraction of samples to be used for fitting the individual base learners, choose from 0<__<1.0
        criterion (string): The function to measure the quality of a split  {‘friedman_mse’, ‘mse’, ‘mae’}
        max_depth (int): maximum depth of the individual regression estimators, defualt =3
    Returns:
        [list]: Model performance listed:  Accuracy, ROC_AUC score, Precision, F1-Score, Confusion Matrix, cross-entropy
    """
  
    model=GradientBoostingClassifier(loss=loss, learning_rate=learning_rate, n_estimators=n_est, \
                                     subsample=subsample, criterion=criterion,max_depth=max_depth)
                
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    ce_loss = log_loss(y_test,y_prob)
    res = model_performance(y_test, y_pred)
    res += [ce_loss]
    return res

def knn (X, y,n_neighbors,weights,algorithm):
    """[K Nearest Neighbor]
    Args:
        X : input data
        y : target
        n_neighbors (int):  Number of neighbors
        weights (string): {‘uniform’, ‘distance’} 
        algorithm (string): {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}
    Returns:
        [list]: Model performance listed:  Accuracy, ROC_AUC score, Precision, F1-Score, Confusion Matrix, cross-entropy
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model=KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights,algorithm=algorithm)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    ce_loss = log_loss(y_test,y_prob)
    res = model_performance(y_test, y_pred)
    res += [ce_loss]

    return res

def Linear_svm(X, y,penalty,loss,C):
    """Linear Support Vector Classification
    Args:
        X : input data
        y : target
        penalty (string): {‘l1’, ‘l2’}
        loss (string): loss function {‘hinge’, ‘squared_hinge’}
        C (float): Regularization parameter, defalut=1.0
    Returns:
        [list]: Model performance listed:  Accuracy, ROC_AUC score, Precision, F1-Score, Confusion Matrix, cross-entropy
    """
    model=LinearSVC(penalty=penalty, loss=loss,C=C)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    ce_loss = log_loss(y_test,y_prob)
    res = model_performance(y_test, y_pred)
    res += [ce_loss]

    return res

def dt_classifier (X,y,criterion, splitter,max_depth):
    """Decision Tree
    Args:
        X ([type]): [description]
        y ([type]): [description]
        criterion (string):  “gini” for the Gini impurity and “entropy” for the information gain.  {“gini”, “entropy”}
        splitter (string):  The strategy used to choose the split at each node {“best”, “random”}
        max_depth (int): The maximum depth of the tree
    Returns:
        [list]: Model performance listed:  Accuracy, ROC_AUC score, Precision, F1-Score, Confusion Matrix, cross-entropy
    """
    model=DecisionTreeClassifier(criterion=criterion,splitter=splitter,max_depth=max_depth)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    ce_loss = log_loss(y_test,y_prob)
    res = model_performance(y_test, y_pred)
    res += [ce_loss]

    return res

def mlp (activation, solver, learning_rate, learning_rate_init):
    """multi-layer perceptron (neutral network)
    Args:
        activation (string  ): {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
        solver (string): {‘lbfgs’, ‘sgd’, ‘adam’}
        learning_rate (string): {‘constant’, ‘invscaling’, ‘adaptive’}
        learning_rate_init (double):  default=0.001
    Returns:
        [list]: Model performance listed:  Accuracy, ROC_AUC score, Precision, F1-Score, Confusion Matrix, cross-entropy
    """
    model=MLPClassifier(activation=activation, solver=sovler, learning_rate=learning_rate,learning_rate_init=learning_rate_init)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    ce_loss = log_loss(y_test,y_prob)
    res = model_performance(y_test, y_pred)
    res += [ce_loss]

    return res