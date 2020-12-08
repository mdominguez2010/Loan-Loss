# Import Dependencies

import numpy as np
import math
from numpy import sort
import pandas as pd
import pickle
import time
import random
from scipy import stats 

import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
sns.set(style='white', palette = 'Paired')
#plt.style.use('ggplot')
%matplotlib inline
%config InlineBackend.figure_formats = ['svg']
np.set_printoptions(suppress=True) # Suppress scientific notation where possible
from ipywidgets import interactive, FloatSlider

from sklearn.inspection import permutation_importance
from sklearn import linear_model, svm, naive_bayes, neighbors, ensemble
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, plot_confusion_matrix, confusion_matrix,\
roc_auc_score, roc_curve, precision_recall_curve, f1_score, fbeta_score, recall_score,\
precision_recall_fscore_support
from sklearn.feature_selection import SelectFromModel

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, make_scorer, log_loss

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import imblearn

from collections import Counter
from mlxtend.plotting import plot_decision_regions

import xgboost as xgb
from xgboost import XGBRegressor

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_error
from sklearn.feature_selection import f_regression, RFE, RFECV

# Prediction Pipeline

class pipeline():
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def test_train_split(self):
        """
        Takes the feature and target data and returns the test_train splits.
        Then, X and y and converted to np.array so it can be used in kf function below
        """
        
        # hold out 20% of the data for final testing
        X, X_test, y, y_test = train_test_split(self.X, self.y, test_size=.2, random_state=42)

        # this helps with the way kf will generate indices below
        X, y = np.array(self.X), np.array(self.y)
        
        return X, X_test, y, y_test
    
    def scale_X_test(self):
        """
        Takes in a dataframe of X_test features and scales them
        """

        # Scale features and test data

        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)

        return X_test_scaled
    
    def xgboost_analysis(self):
        """
        Takes in a dataframe of X features and a y vector of targets and runs XGBoost with over and undersampling.
        10-fold cross-validation is used.
        Prints scores for validate and test data
        Returns both models, gbm_under and gbm_over.
        """

        kf = KFold(n_splits=10, shuffle=True, random_state = 42)
        gbm_under_scores, gbm_over_scores = [],[]

        ros = RandomOverSampler(random_state=42)
        rus = RandomUnderSampler(random_state=42)

        for train_ind, val_ind in kf.split(X,y):

            X_train, y_train = X[train_ind], y[train_ind]
            X_val, y_val = X[val_ind], y[val_ind]

            # Create undersampling of the data
            X_train_under, y_train_under = rus.fit_sample(X_train, y_train)
            # Create oversampling of the data
            X_train_over, y_train_over = ros.fit_sample(X_train, y_train)

            # Scale the features
            scaler_under = StandardScaler()
            X_train_under_scaled = scaler_under.fit_transform(X_train_under)
            X_val_scaled = scaler_under.transform(X_val)

            scaler_over = StandardScaler()
            X_train_over_scaled = scaler_over.fit_transform(X_train_over)
            X_val_scaled = scaler_over.transform(X_val)

            # XGBoost - Undersampling
            gbm_under = xgb.XGBClassifier(n_estimators=150, min_child_weight=7, max_depth=3, gamma=0.0, colsample_bytree=0.5)
            gbm_under.fit(X_train_under_scaled, y_train_under)
            gbm_under_scores.append(gbm_under.score(X_val_scaled,y_val))

            # XGBoost Forest - Oversampling
            gbm_over = xgb.XGBClassifier(n_estimators=150, min_child_weight=7, max_depth=3, gamma=0.0, colsample_bytree=0.5)
            gbm_over.fit(X_train_over_scaled, y_train_over)
            gbm_over_scores.append(gbm_over.score(X_val_scaled, y_val))

        gbm_val_under_score = round(np.mean(gbm_under_scores),4)
        gbm_val_over_score = round(np.mean(gbm_over_scores),4)

        print('Calculating scores...\n')
        print(f'XGBoost undersampling val score: {gbm_val_under_score}')
        print(f'XGBoost undersampling test score: {gbm_under.score(X_test_scaled,y_test)}')
        print('\n')
        print(f'XGBoost oversampling val score: {gbm_val_over_score}')
        print(f'XGBoost oversampling test score: {gbm_over.score(X_test_scaled,y_test)}')

        print('\nCalculating Classification Reports...\n')

        print('Undersampling Method\n')
        y_preds_gbm_under = gbm_under.predict(X_test_scaled)
        print(classification_report(y_test, y_preds_gbm_under))

        print('Oversampling Method\n')
        y_preds_gbm_over = gbm_over.predict(X_test_scaled)
        print(classification_report(y_test, y_preds_gbm_over))

        print('\nCalculating F-beta scores...\n')

        print(f"Fbeta score Undersampled: {fbeta_score(y_test, y_preds_gbm_under, average=None, beta=10.0)}")
        print(f"Fbeta score Oversampled: {fbeta_score(y_test, y_preds_gbm_over, average=None, beta=10.0)}")

        print('\nCalculating F1 scores...\n')

        print(f"F1 score Undersampled: {f1_score(y_test, y_preds_gbm_under, average='weighted')}")
        print(f"F1 score Oversampled: {f1_score(y_test, y_preds_gbm_over, average='weighted')}")

        print('\nCalculating Confusion Matrices...\n')

        titles_models = [("XGBoost - Undersampling Method", gbm_under), ("XGBoost - Oversampling Method", gbm_over)]

        for title, model in titles_models:

            fig, ax = plt.subplots(figsize=(7, 7))
            disp = plot_confusion_matrix(model, X_test_scaled, y_test, ax=ax)
            disp.ax_.set_title(title)
            print(title)
            print(disp.confusion_matrix)

        return gbm_under, gbm_over
    
    def plot_roc_curve(self, model):
        """
        Takes in a model and returns the ROC curve and score
        """
        fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test_scaled)[:,1])

        plt.plot(fpr, tpr,lw=2)
        plt.plot([0,1],[0,1],c='violet',ls='--')
        plt.xlim([-0.05,1.05])
        plt.ylim([-0.05,1.05])

        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve');

        print("ROC AUC score = ", roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:,1]))
    
    def n_risky(self, model):
        """
        Returns the number of risky loans our model predicted
        """
        n_risky = 0
        for prediction in model.predict(X_test_scaled):
            if prediction == 1:
                n_risky += 1
        
        return n_risky
    
    def risky_loans(self, model):
        """
        Create dataframe of risky loans to be used in linear regression model
        """

        # Insert the predicted 'Risky?' column
        X_test['Risky?'] = model.predict(X_test_scaled)

        # Extract indices for loans predicted to have some loss
        risky_indices = X_test[X_test['Risky?'] == 1].index

        # Create dataframe
        risky_loans = train_regression.iloc[risky_indices]
        
        # Add a transformation column log_loss to be used in linear regression
        risky_loans['log_loss'] = np.log(risky_loans['loss'] + 1)

        return risky_loans
    
    def regression_analysis(self):
        """
        Conducts regression anlaysls. Prints R-squared and MAE for each algorithm
        """
        kf = KFold(n_splits=5, shuffle=True, random_state = 42)
        lr_scores, lr_ridge_scores, lr_lasso_scores, \
        lr_elastic_scores, lr_xgb_scores = [], [], [], [], [] #collect the validation results for all models

        for train_ind, val_ind in kf.split(X,y):

            X_train, y_train = X[train_ind], y[train_ind]
            X_val, y_val = X[val_ind], y[val_ind] 

            # scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # load models
            lr = LinearRegression()
            lr_ridge = RidgeCV(cv=5)
            lr_lasso = LassoCV(cv=5, tol=0.5, max_iter=50000)
            lr_elastic = ElasticNetCV(cv=5, random_state=42, tol=0.5, max_iter=50000)
            lr_xgb = XGBRegressor(n_estimators=150, min_child_weight=7, max_depth=3, gamma=0.0, colsample_bytree=0.5)

            # fit models
            lr.fit(X_train_scaled, y_train)
            lr_ridge.fit(X_train_scaled, y_train)
            lr_lasso.fit(X_train_scaled, y_train)
            lr_elastic.fit(X_train_scaled, y_train)
            lr_xgb.fit(X_train_scaled, y_train)

            # create lists of scores
            lr_scores.append(lr.score(X_val_scaled, y_val))
            lr_ridge_scores.append(lr_ridge.score(X_val_scaled, y_val))
            lr_lasso_scores.append(lr_lasso.score(X_val_scaled, y_val))
            lr_elastic_scores.append(lr_elastic.score(X_val_scaled, y_val))
            lr_xgb_scores.append(lr_xgb.score(X_val_scaled, y_val))

        # Test
        
        # Prints all of our metrics
        
        print(f'Linear Regression val R^2: {np.mean(lr_scores):.3f} +- {np.std(lr_scores):.3f}')
        print(f'Linear Regression test R^2: {lr.score(X_test_scaled, y_test):.3f}')
        print(f'MAE: {mean_absolute_error(y_test, lr.predict(X_test_scaled)):.3f}\n')

        print(f'Ridge Regression val R^2: {np.mean(lr_ridge_scores):.3f} +- {np.std(lr_ridge_scores):.3f}')
        print(f'Ridge Regression test R^2: {lr_ridge.score(X_test_scaled, y_test):.3f}')
        print(f'MAE: {mean_absolute_error(y_test, lr_ridge.predict(X_test_scaled)):.3f}\n')

        print(f'Lasso Regression val R^2: {np.mean(lr_lasso_scores):.3f} +- {np.std(lr_lasso_scores):.3f}')
        print(f'Lasso Regression test R^2: {lr_lasso.score(X_test_scaled, y_test):.3f}')
        print(f'MAE: {mean_absolute_error(y_test, lr_lasso.predict(X_test_scaled)):.3f}\n')

        print(f'Elastic Net Regression val R^2: {np.mean(lr_elastic_scores):.3f} +- {np.std(lr_elastic_scores):.3f}')
        print(f'Elastic Net Regression test R^2: {lr_elastic.score(X_test_scaled, y_test):.3f}')
        print(f'MAE: {mean_absolute_error(y_test, lr_elastic.predict(X_test_scaled)):.3f}\n')

        print(f'XGBoost Regression val R^2: {np.mean(lr_xgb_scores):.3f} +- {np.std(lr_xgb_scores):.3f}')
        print(f'XGBoost Regression test R^2: {lr_xgb.score(X_test_scaled, y_test):.3f}')
        print(f'MAE: {mean_absolute_error(y_test, lr_xgb.predict(X_test_scaled)):.3f}')
        
        # Put models and respective scores in a list
        
        model_score_list = [[lr, lr.score(X_test_scaled, y_test)],
                   [lr_ridge, lr_ridge.score(X_test_scaled, y_test)],
                   [lr_lasso, lr_lasso.score(X_test_scaled, y_test)],
                   [lr_elastic, lr_elastic.score(X_test_scaled, y_test)],
                   [lr_xgb, lr_xgb.score(X_test_scaled, y_test)]]
        
        model_score_list = sorted(model_score_list, key = lambda x: x[1])
        
        # Return model with highest R-squared
        
        return model_score_list[-1][0]
    
    def plot_residuals(self, regression_model):
        """
        Residuals v Predicted plot
        """

        residuals = y_test - regression_model.predict(X_test_scaled)

        plt.figure(figsize=(10, 7))
        plt.scatter(regression_model.predict(X_test_scaled), residuals)   

        plt.axhline(0, linestyle='--', color='gray')
        plt.xlabel('Predicted Values', fontsize=18)
        plt.ylabel('Residuals', fontsize=18);
    
    def predictions(self, X_test):
        """
        Returns loss predictions and back-transforms to original scale
        """
        # Predictions
        y_pred_log = regression_model.predict(X_test)
        
        # Predictions back-transformed        
        y_pred = np.array([math.exp(x)-1 for x in y_pred_log])

        return y_pred
    
    def plot_dist(self):
        """
        Plot distributions of loan loss predictions
        """
        
        # Distribution of predicted loan losses
        ax = sns.histplot(y_pred, bins=30, log_scale=True, stat='probability', kde=True)
        ax.set(xlabel='loss')
        plt.xlim(0,100);