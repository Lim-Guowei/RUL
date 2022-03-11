import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import json
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

def predict(filename, result_dir):

    _, df_test = dataloader(filename)
    df_test = df_test.drop(columns=["fan_eff_mod", "fan_flow_mod", "LPC_eff_mod", "LPC_flow_mod", "HPC_eff_mod", "HPC_flow_mod", 
                                    "HPT_flow_mod", "LPT_eff_mod", "LPT_flow_mod", "cycle", "unit"])
    Y_true = df_test["RUL"].values    
    df_test_features = df_test.drop(["RUL"], axis=1)
    clf = joblib.load(os.path.normpath(os.path.join(result_dir, "model.pkl")))
    
    startTime = time.time()
    Y_pred = clf.predict(df_test_features)
    elapsedTime = time.time() - startTime
    rmse = math.sqrt(mean_squared_error(Y_true, Y_pred))
    print("RMSE is {:.4f}".format(rmse))
    print(clf.named_steps["model"].get_params())

    with open(os.path.normpath(os.path.join(result_dir, "predictionReport.txt")), "w+") as f:
        f.write("RMSE using best estimator is {:.4f}\n".format(rmse))
        f.write(json.dumps(clf.named_steps["model"].get_params()))

    df_test["RUL_predicted"] = Y_pred

    fig = df_test.plot(y=["RUL", "RUL_predicted"], 
                        kind="line", 
                        title="Ground truth vs Predicted for N-CMAPSS_DS01-005", 
                        xlabel="index", 
                        use_index=True,
                        linewidth=1.0,
                        alpha=0.7,
                        xlim=(0, df_test.index.max()),
                        figsize=(20, 15)
                        ).get_figure()

    fig.savefig(os.path.normpath(os.path.join(result_dir, "predict.png")))
    return

def predict_on_dev(filename, result_dir):
    df_dev, _ = dataloader(filename)
    df_dev = df_dev.drop(columns=["fan_eff_mod", "fan_flow_mod", "LPC_eff_mod", "LPC_flow_mod", "HPC_eff_mod", "HPC_flow_mod", 
                                    "HPT_flow_mod", "LPT_eff_mod", "LPT_flow_mod", "cycle", "unit"])

    Y_true = df_dev["RUL"].values    
    df_dev_features = df_dev.drop(["RUL"], axis=1)
    clf = joblib.load(os.path.normpath(os.path.join(result_dir, "model.pkl")))
    
    startTime = time.time()
    Y_pred = clf.predict(df_dev_features)
    elapsedTime = time.time() - startTime
    rmse = math.sqrt(mean_squared_error(Y_true, Y_pred))
    print("RMSE is {:.4f}".format(rmse))
    print(clf.named_steps["model"].get_params())

    with open(os.path.normpath(os.path.join(result_dir, "predictionReport_on_dev.txt")), "w+") as f:
        f.write("RMSE using best estimator is {:.4f}\n".format(rmse))
        f.write(json.dumps(clf.named_steps["model"].get_params()))

    df_dev["RUL_predicted"] = Y_pred

    fig = df_dev.plot(y=["RUL", "RUL_predicted"], 
                        kind="line", 
                        title="Ground truth vs Predicted for N-CMAPSS_DS01-005", 
                        xlabel="index", 
                        use_index=True,
                        linewidth=1.0,
                        alpha=0.7,
                        xlim=(0, df_dev.index.max()),
                        figsize=(20, 15)
                        ).get_figure()

    fig.savefig(os.path.normpath(os.path.join(result_dir, "predict_on_dev.png")))
    return

if __name__ == "__main__":
    # predict("N-CMAPSS_DS01-005.h5", "xgbregressor")
    # predict("N-CMAPSS_DS01-005.h5", "randomforestregressor")
    # predict("N-CMAPSS_DS01-005.h5", "ridgeregressor")

    # predict_on_dev("N-CMAPSS_DS01-005.h5", "xgbregressor")
    predict_on_dev("N-CMAPSS_DS01-005.h5", "randomforestregressor")
    predict_on_dev("N-CMAPSS_DS01-005.h5", "ridgeregressor")
