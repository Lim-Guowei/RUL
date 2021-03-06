
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from dataloader import dataloader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import joblib
import time
from scipy.stats import reciprocal, randint, uniform
from math import ceil
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

class FeatureSelector(BaseEstimator, TransformerMixin):
    """ To select dataframe columns for Pipeline"""
    # Class Constructor
    def __init__(self, feature_names):
        self.feature_names = feature_names

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self
    
    # Method that describes what we need this transformer to do
    def transform(self, X, y=None):
        if  self.feature_names:
            return X[self.feature_names] 

def model(filename, result_dir, add_lag=False):

    cwd = os.getcwd()
    os.makedirs(os.path.normpath(os.path.join(cwd, result_dir)), exist_ok=True)

    # From EDA, the following information are determined:
    # - No null values in dataset
    # - Remove columns ["fan_eff_mod", "fan_flow_mod", "LPC_eff_mod", "LPC_flow_mod", "HPC_eff_mod", "HPC_flow_mod", "HPT_flow_mod", 
    #   "LPT_eff_mod", "LPT_flow_mod"] as they contain all zeroes values
    # - Remove column "cycle" as "RUL" is strongly correlated and is derived from it
    # - Remove column "unit" as test data does not contain units from dev data
    # - Adding lag 1, 3 and 5 features on RUL value and treat time series as a regressive problem
    # - ["Fc", "hs"] are categorical features
    
    # Pre-processing based on EDA findings
    df_dev, _ = dataloader(filename)
    df_dev = df_dev.drop(columns=["fan_eff_mod", "fan_flow_mod", "LPC_eff_mod", "LPC_flow_mod", "HPC_eff_mod", "HPC_flow_mod", "HPT_flow_mod", "LPT_eff_mod", "LPT_flow_mod", "cycle", "unit"])

    if add_lag:
        df_dev["RUL_lag1"] = df_dev["RUL"].shift(1)
        df_dev["RUL_lag3"] = df_dev["RUL"].shift(3)
        df_dev["RUL_lag5"] = df_dev["RUL"].shift(5)
        df_dev = df_dev.iloc[5::] # Discard NaN rows

    # Model training
    Y = df_dev["RUL"]
    X = df_dev.drop(["RUL"], axis=1)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=True, test_size=0.2)
    
    # categoricalFeatures = ["unit", "Fc", "hs"]
    categoricalFeatures = ["Fc", "hs"]
    continuousFeatures = [x for x in X.columns.tolist() if x not in categoricalFeatures]

    categoricalPipeline = Pipeline(steps=[
                                     ("categoricalSelector", FeatureSelector(feature_names=categoricalFeatures)),
                                     ("oneHotEncoder", OneHotEncoder(handle_unknown='ignore', sparse=True, dtype=np.int)) # Labelencoder is already included in OneHotEncoder with new update, # Set spare=False for GaussianNB model only, else True
                                     ])

    continuousPipeline = Pipeline(steps=[
                                        ("continuousSelector", FeatureSelector(feature_names=continuousFeatures)), 
                                        ('scaler', StandardScaler()),
                                        ('pca', PCA(n_components=0.90)) #retain components that explain 90% of the variance
                                        ])

    unionPipeline = FeatureUnion(transformer_list=[
                                                    ("continuousPipeline", continuousPipeline),
                                                    ("categoricalPipeline", categoricalPipeline)
                                                    ])

    if result_dir in ["xgbregressor", "xgbregressor_lag"]:
        mainPipeline = Pipeline(steps=[
                                        ("mainPipeline", unionPipeline),
                                        ("model", XGBRegressor())   # Change model according to required model imported as shown in first few lines, # Remove random_state and verbose for GaussianNB
                                        ])
    
    elif result_dir in ["randomforestregressor", "randomforestregressor_lag"]:
        mainPipeline = Pipeline(steps=[
                                        ("mainPipeline", unionPipeline),
                                        ("model", RandomForestRegressor())   # Change model according to required model imported as shown in first few lines, # Remove random_state and verbose for GaussianNB
                                        ])

    elif result_dir in ["ridgeregressor", "ridgeregressor_lag"]:
        mainPipeline = Pipeline(steps=[
                                ("mainPipeline", unionPipeline),
                                ("model", Ridge())   # Change model according to required model imported as shown in first few lines, # Remove random_state and verbose for GaussianNB
                                ])
    else:
        sys.exit("No model available")

    paramsXGBoostRegressor = {
                                'model__n_estimators': [100],
                                'model__max_depth': randint(1, 2),
                                'model__subsample': uniform(0.25, 0.75),
                                'model__colsample_bytree': uniform(0.25, 0.75)
    }

    paramsRandomForestRegressor = {
                                    'model__n_estimators': randint(1e1, 1e2),
                                    'model__min_samples_leaf': randint(1e0, 2e0),
                                    'model__max_features': ['auto'], 
                                    'model__max_depth': randint(1e0, 4e0),
                                    'model__min_samples_split': randint(2e0, 4e0)
                                    }    

    paramsRidgeRegressor = {
                            'model__alpha': uniform(0.1, 10.0)
                            }

    startTime = time.time()
    if result_dir in ["xgbregressor", "xgbregressor_lag"]:
        clf = RandomizedSearchCV(mainPipeline, param_distributions=paramsXGBoostRegressor, cv=5, scoring="neg_mean_squared_error", n_jobs=2 ,verbose=10, n_iter=3) # change param_distributions parameter according to fitted model
    
    elif result_dir in ["randomforestregressor", "randomforestregressor_lag"]:
        clf = RandomizedSearchCV(mainPipeline, param_distributions=paramsRandomForestRegressor, cv=5, scoring="neg_mean_squared_error", n_jobs=2 ,verbose=10, n_iter=3) # change param_distributions parameter according to fitted model
    
    elif result_dir in ["ridgeregressor", "ridgeregressor_lag"]:
        clf = RandomizedSearchCV(mainPipeline, param_distributions=paramsRidgeRegressor, cv=5, scoring="neg_mean_squared_error", n_jobs=2 ,verbose=10, n_iter=3) # change param_distributions parameter according to fitted model
    
    else:
        sys.exit("No model available")
    
    clf.fit(X_train, Y_train) 
    elapsedTime = time.time() - startTime
    cvTestresults = pd.DataFrame(clf.cv_results_)
    cvTestresults.to_csv(os.path.normpath(os.path.join(result_dir, "cvTestresults.csv")), sep=";", header=True, index=False)
    print(clf.best_estimator_)

    # Save model and printout
    joblib.dump(clf.best_estimator_, os.path.normpath(os.path.join(result_dir, "model.pkl")), compress=1)    # Save estimator
    clf = joblib.load(os.path.normpath(os.path.join(result_dir, "model.pkl")))       # Load estimator
    score = math.sqrt(abs(clf.score(X_test, Y_test)))
    print("RMSE using best estimator is {:.4f}".format(score))
    print("Completed. Time taken is {:.4f}s".format(elapsedTime))

    with open(os.path.normpath(os.path.join(result_dir, "modelReport.txt")), "w+") as f:
        f.write("RMSE using best estimator is {:.4f}\n".format(score))
        f.write("Time taken for computation is {:.4f}s".format(elapsedTime))
    return

if __name__ == "__main__":

    model("N-CMAPSS_DS01-005.h5", "xgbregressor")      
    model("N-CMAPSS_DS01-005.h5", "randomforestregressor")
    model("N-CMAPSS_DS01-005.h5", "ridgeregressor")                 

    model("N-CMAPSS_DS01-005.h5", "xgbregressor_lag", add_lag=True)      
    model("N-CMAPSS_DS01-005.h5", "randomforestregressor_lag", add_lag=True)
    model("N-CMAPSS_DS01-005.h5", "ridgeregressor_lag", add_lag=True)                 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         