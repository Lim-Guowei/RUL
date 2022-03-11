import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataloader import dataloader
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

pd.set_option('display.float_format', '{:.6f}'.format)

def countNullPercent(dataframe):
    """ Print percentage of null values for each column in dataframe sorted in descending order
    """
    nullCollect = {}
    for column in dataframe:
        rowCount = len(dataframe[column])
        nullCount = dataframe[column].isnull().sum()
        percentNull = round((nullCount/rowCount)*100, 2)
        nullCollect.update({column: percentNull})

    for key, value in sorted(nullCollect.items(), key=lambda item: item[1], reverse=True):  # Sort dictionary based on value in descending order
        print("{}: {}".format(key, value))
    return     

def countUniqueVal(dataframe, column):
    """ Print unique values for each columns
    """
    for count, name in enumerate(column):
        print("#{} - {}".format(count, name))
        print(dataframe[name].value_counts())
        print("\n")
    return

def plot_by_unit(dataframe, unit):
    """ Generate visualization for each fleet unit
        Unit number can be obtained by inspecting "unit" column in dataframe
        Generate plot for each variable (x-axis) vs rul (y-axis)
    """
    df_unit = dataframe[dataframe["unit"] == unit]
    print(df_unit)

    ### Correlation plot
    plt.subplots(figsize=(20,15))
    color = plt.get_cmap('inferno')   # default color
    color.set_bad('lightblue')
    corr_plot = sns.heatmap(data=df_unit.corr(), annot=False, cmap=color)
    plt.title("Correlation matrix for unit {}".format(unit), fontdict={'fontsize': 16})
    plt.savefig("corr_plot_unit_{}.png".format(unit))
    return

def rank_feature_importance(dataframe):
    feat_labels = dataframe.columns.values

    Y = dataframe["RUL"]
    X = dataframe.drop(["RUL"], axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, shuffle=True, test_size=0.2)
    
    # Create a random forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)

    # Train the classifier
    clf.fit(X_train, Y_train)

    # Plot random forest feature importance
    importances = clf.feature_importances_
    indices = np.argsort(importances)

    plt.title('Feature Importances', fontdict={'fontsize': 16})
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feat_labels[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.savefig("feature_importance.png")
    return

def add_lag_features(dataframe):

    dataframe["RUL_lag1"] = dataframe["RUL"].shift(1)
    dataframe["RUL_lag3"] = dataframe["RUL"].shift(3)
    dataframe["RUL_lag5"] = dataframe["RUL"].shift(5)
    dataframe = dataframe.iloc[5::] # Discard NaN rows
    
    fig = dataframe.plot(y=["RUL", "RUL_lag1", "RUL_lag1", "RUL_lag3", "RUL_lag5"], 
                            kind="line", 
                            title="Lag on RUL variable", 
                            xlabel="index", 
                            use_index=True,
                            linewidth=1.0,
                            alpha=0.7,
                            xlim=(0, dataframe.index.max()),
                            figsize=(20, 15)
                            ).get_figure()
    
    fig.savefig("lag_on_RUL.png")
    return

def eda(filename):
    df_dev, df_test = dataloader(filename)
    column_name = df_dev.columns.tolist()

    ### Check for null or zeroes
    countNullPercent(df_dev)    # No null values in dataframe
    countNullPercent(df_test)   # No null values in dataframe
    df_dev.describe().to_csv("df_dev_description.csv")
    df_test.describe().to_csv("df_test_description.csv")

    # Remove columns containing all zeroes
    # Remove "cycle" as "RUL" is sufficient as target variable
    df_dev = df_dev.drop(columns=["fan_eff_mod", "fan_flow_mod", "LPC_eff_mod", "LPC_flow_mod", "HPC_eff_mod", "HPC_flow_mod", "HPT_flow_mod", "LPT_eff_mod", "LPT_flow_mod", "cycle"])
    df_test = df_test.drop(columns=["fan_eff_mod", "fan_flow_mod", "LPC_eff_mod", "LPC_flow_mod", "HPC_eff_mod", "HPC_flow_mod", "HPT_flow_mod", "LPT_eff_mod", "LPT_flow_mod", "cycle"])

    ### Identify categorical features as "unit", "Fc", "hs"
    countUniqueVal(df_dev, ["unit", "Fc", "hs"])

    ### Generate correlation matrix plot for each unit in fleet 
    plot_by_unit(df_dev, 1.0)
    plot_by_unit(df_dev, 2.0)
    plot_by_unit(df_dev, 3.0)
    plot_by_unit(df_dev, 4.0)
    plot_by_unit(df_dev, 5.0)
    plot_by_unit(df_dev, 6.0)

    # Rank feature importance using random forest classifier
    rank_feature_importance(df_dev)

    add_lag_features(df_dev)

    return
    
if __name__ == "__main__":
    eda("N-CMAPSS_DS01-005.h5")