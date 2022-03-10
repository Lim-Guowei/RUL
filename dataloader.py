import os
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def dataloader(filename):
    dirname = "Dataset"
    filepath = os.path.normpath(os.path.join(os.path.join(os.getcwd(), dirname), filename))

    with h5py.File(filepath, "r") as hdf:
        print("Keys: {}".format(hdf.keys()))
        # Development set
        W_dev = np.array(hdf.get('W_dev'))             # W  - operative condition
        X_s_dev = np.array(hdf.get('X_s_dev'))         # X_s - measured signal
        X_v_dev = np.array(hdf.get('X_v_dev'))         # X_v - virtual sensors
        T_dev = np.array(hdf.get('T_dev'))             # T - engine health parameters
        Y_dev = np.array(hdf.get('Y_dev'))             # RUL - RUL label
        A_dev = np.array(hdf.get('A_dev'))             # Auxiliary - unit number u and the flight cycle number c, the flight class Fc and the health state h s

        # Test set
        W_test = np.array(hdf.get('W_test'))           # W
        X_s_test = np.array(hdf.get('X_s_test'))       # X_s
        X_v_test = np.array(hdf.get('X_v_test'))       # X_v
        T_test = np.array(hdf.get('T_test'))           # T
        Y_test = np.array(hdf.get('Y_test'))           # RUL  
        A_test = np.array(hdf.get('A_test'))           # Auxiliary
        
        # Varnams
        W_var = np.array(hdf.get('W_var'))
        X_s_var = np.array(hdf.get('X_s_var'))  
        X_v_var = np.array(hdf.get('X_v_var')) 
        T_var = np.array(hdf.get('T_var'))
        A_var = np.array(hdf.get('A_var'))

        # from np.array to list dtype U4/U5
        W_var = list(np.array(W_var, dtype='U20'))
        X_s_var = list(np.array(X_s_var, dtype='U20'))  
        X_v_var = list(np.array(X_v_var, dtype='U20')) 
        T_var = list(np.array(T_var, dtype='U20'))
        A_var = list(np.array(A_var, dtype='U20'))

    # W = np.concatenate((W_dev, W_test), axis=0)  
    # X_s = np.concatenate((X_s_dev, X_s_test), axis=0)
    # X_v = np.concatenate((X_v_dev, X_v_test), axis=0)
    # T = np.concatenate((T_dev, T_test), axis=0)
    # Y = np.concatenate((Y_dev, Y_test), axis=0) 
    # A = np.concatenate((A_dev, A_test), axis=0) 

    dev_data = np.concatenate((W_dev, X_s_dev, X_v_dev, T_dev, A_dev, Y_dev), axis=1)
    test_data = np.concatenate((W_test, X_s_test, X_v_test, T_test, A_test, Y_test), axis=1)
    column_name = W_var + X_s_var + X_v_var + T_var + A_var
    column_name.append("RUL")

    # print("dev_data shape: {}".format(dev_data.shape))
    # print("test_data shape: {}".format(test_data.shape))
    # print("column_name shape: {}".format(len(column_name)))
    print("column_name: {}".format(column_name))
    
    # print ("W shape: " + str(W.shape))
    # print ("X_s shape: " + str(X_s.shape))
    # print ("X_v shape: " + str(X_v.shape))
    # print ("T shape: " + str(T.shape))
    # print("Y shape: " + str(Y.shape))
    # print ("A shape: " + str(A.shape))
    # print("Variables in W_var: {}".format(W_var))
    # print("Variables in X_s_var: {}".format(X_s_var))
    # print("Variables in X_v_var: {}".format(X_v_var))
    # print("Variables in T_var: {}".format(T_var))
    # print("Variables in A_var: {}".format(A_var))

    df_dev = pd.DataFrame(data=dev_data, columns=column_name)
    df_test = pd.DataFrame(data=test_data, columns=column_name)
    # df_dev.iloc[100000: 100100].to_csv("df_dev_100.csv") 
    return df_dev, df_test

if __name__ == "__main__":
    dataloader("N-CMAPSS_DS01-005.h5")