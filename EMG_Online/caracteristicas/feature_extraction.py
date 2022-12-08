import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
from IPython.display import clear_output
import json
from statistics import mode

def MAV_EMG(df, WINDOW_SIZE):
    MAV_values = []
    for j in range(8):
        MAV_temp = []
        for i in range(1,max(df['signal']) + 1):
            if(len(df[df['signal'] == i]) >= WINDOW_SIZE):
                samples_temp = df[df['signal'] == i][:WINDOW_SIZE]['EMG_s'+ str(j)].reset_index(drop=True)
                MAV_temp.append(mean(abs(samples_temp)))
        MAV_values.append(MAV_temp)
    return MAV_values

def WL_EMG(df, WINDOW_SIZE):
    WL_values = []
    for j in range(8):
        WL_temp = []
        for i in range(1,max(df['signal']) + 1):
            if(len(df[df['signal'] == i]) >= WINDOW_SIZE):
                samples_temp = df[df['signal'] == i][:WINDOW_SIZE-1]['EMG_s'+ str(j)].reset_index(drop=True)
                samples_temp_plus = df[df['signal'] == i][1:WINDOW_SIZE]['EMG_s'+ str(j)].reset_index(drop=True)
                WL_temp.append(sum(abs(samples_temp_plus-samples_temp)))
        WL_values.append(WL_temp)
    return WL_values

def SSC_EMG(df, WINDOW_SIZE, limiar):
    SSC_values = []
    for j in range(8):
        SSC_temp = []
        for i in range(1,max(df['signal']) + 1):
            if(len(df[df['signal'] == i]) >= WINDOW_SIZE):
                samples_temp = df[df['signal'] == i]['EMG_s'+ str(j)][1:WINDOW_SIZE-1].reset_index(drop=True)
                samples_temp_plus = df[df['signal'] == i]['EMG_s'+ str(j)][2:WINDOW_SIZE].reset_index(drop=True)
                samples_temp_minus = df[df['signal'] == i]['EMG_s'+ str(j)][:WINDOW_SIZE-2].reset_index(drop=True)
                SSC_temp.append(sum(((samples_temp - samples_temp_minus)*(samples_temp - samples_temp_plus)>limiar).astype(int)))
        SSC_values.append(SSC_temp)
    return SSC_values

def ZC_EMG(df, WINDOW_SIZE, limiar):
    ZC_values = []
    for j in range(8):
        ZC_temp = []
        for i in range(1,max(df['signal']) + 1):
            if(len(df[df['signal'] == i]) >= WINDOW_SIZE):
                samples_temp = df[df['signal'] == i]['EMG_s'+ str(j)][:WINDOW_SIZE-1].reset_index(drop=True)
                samples_temp_plus = df[df['signal'] == i]['EMG_s'+ str(j)][1:WINDOW_SIZE].reset_index(drop=True)
                op1 = (samples_temp*samples_temp_plus) >= limiar
                op2 = (samples_temp - samples_temp_plus) >= limiar
                ZC_temp.append(sum(op1!=op2))
        ZC_values.append(ZC_temp)
    return ZC_values

#--------------------------------------------------------------------------------------------------------------------------

def IEMG_EMG(df, WINDOW_SIZE):
    IEMG_values = []
    for j in range(8):
        IEMG_temp = []
        for i in range(1,max(df['signal']) + 1):
            if(len(df[df['signal'] == i]) >= WINDOW_SIZE):
                samples_temp = df[df['signal'] == i]['EMG_s'+ str(j)][:WINDOW_SIZE].reset_index(drop=True)
                IEMG_temp.append(sum(abs(samples_temp)))
        IEMG_values.append(IEMG_temp)
    return IEMG_values

def DASDV_EMG(df, WINDOW_SIZE):
    DASDV_values = []
    for j in range(8):
        DASDV_temp = []
        for i in range(1,max(df['signal']) + 1):
            if(len(df[df['signal'] == i]) >= WINDOW_SIZE):
                samples_temp = df[df['signal'] == i]['EMG_s'+ str(j)][:WINDOW_SIZE-1].reset_index(drop=True)
                samples_temp_plus = df[df['signal'] == i]['EMG_s'+ str(j)][1:WINDOW_SIZE].reset_index(drop=True)
                DASDV_temp.append(sum(pow(samples_temp_plus-samples_temp,2))/len(samples_temp))
                if (np.isnan(sum(pow(samples_temp_plus-samples_temp,2))/len(samples_temp))):
                    print(len(samples_temp))
                    print(len(samples_temp_plus))
                    print(len(df[df['signal'] == i]))
                    print()
        DASDV_values.append(DASDV_temp)
    return DASDV_values

def WAMP_EMG(df, WINDOW_SIZE, limiar):
    WAMP_values = []
    for j in range(8):
        WAMP_temp = []
        for i in range(1,max(df['signal']) + 1):
            if(len(df[df['signal'] == i]) > WINDOW_SIZE):
                samples_temp = df[df['signal'] == i]['EMG_s'+ str(j)][:WINDOW_SIZE-1].reset_index(drop=True)
                samples_temp_plus = df[df['signal'] == i]['EMG_s'+ str(j)][1:WINDOW_SIZE].reset_index(drop=True)
                WAMP_temp.append(sum(abs(samples_temp-samples_temp_plus)>=limiar))
        WAMP_values.append(WAMP_temp)
    return WAMP_values

def AR4_EMG(df, WINDOW_SIZE):
    AR4_values = []
    for j in range(8):
        AR4_temp = []
        for i in range(1,max(df['signal']) + 1):
            if(len(df[df['signal'] == i]) >= WINDOW_SIZE):
                samples_temp = df[df['signal'] == i]['EMG_s'+ str(j)][:WINDOW_SIZE].reset_index(drop=True)
                rho, sigma = sm.regression.linear_model.burg(samples_temp, order=4)
                AR4_temp.append(rho.tolist())
        AR4_values.append(AR4_temp)
    return AR4_values

def VORDER_EMG(df, WINDOW_SIZE, v):
    VORDER_values = []
    for j in range(8):
        VORDER_temp = []
        for i in range(1,max(df['signal']) + 1):
            if(len(df[df['signal'] == i]) >= WINDOW_SIZE):
                samples_temp = df[df['signal'] == i]['EMG_s'+ str(j)][:WINDOW_SIZE].reset_index(drop=True)
                VORDER_temp.append(abs(pow(sum(pow(samples_temp,v))/len(samples_temp),1/v)))
        VORDER_values.append(VORDER_temp)
    return VORDER_values

#--------------------------------------------------------------------------------------------------------------------------

def VAR_EMG(df, WINDOW_SIZE):
    VAR_values = []
    for j in range(8):
        VAR_temp = []
        for i in range(1,max(df['signal']) + 1):
            if(len(df[df['signal'] == i]) > WINDOW_SIZE):
                samples_temp = df[df['signal'] == i]['EMG_s'+ str(j)][:WINDOW_SIZE].reset_index(drop=True)
                VAR_temp.append(sum(pow(samples_temp,2))/(len(samples_temp)-1))
        VAR_values.append(VAR_temp)
    return VAR_values

def RMS_EMG(df, WINDOW_SIZE):
    RMS_values = []
    for j in range(8):
        RMS_temp = []
        for i in range(1,max(df['signal']) + 1):
            if(len(df[df['signal'] == i]) > WINDOW_SIZE):
                samples_temp = df[df['signal'] == i]['EMG_s'+ str(j)][:WINDOW_SIZE].reset_index(drop=True)
                RMS_temp.append(pow(sum(pow(samples_temp,2))/len(samples_temp),1/2))
        RMS_values.append(RMS_temp)
    return RMS_values
	
def GROUP_1 (df, WINDOW_SIZE, limiar, label):
    # MAV, WL, SSC E ZC
    MAV_LIST = MAV_EMG(df, WINDOW_SIZE)
    WL_LIST  = WL_EMG(df, WINDOW_SIZE)
    SSC_LIST = SSC_EMG(df, WINDOW_SIZE,limiar)
    ZC_LIST  = ZC_EMG(df, WINDOW_SIZE,limiar)
    dict_list = []
    for j in range(min([max(df['signal']), 10])):
        if(len(df[df['signal'] == j+1]) >= WINDOW_SIZE):
            entry = [label]
            sensor_list = []
            for i in range(8):
                sensor_list.append(MAV_LIST[i][j])
                sensor_list.append(WL_LIST[i][j])
                sensor_list.append(SSC_LIST[i][j])
                sensor_list.append(ZC_LIST[i][j])
                if(np.isnan(MAV_LIST[i][j])):
                    print('MAV')
                if(np.isnan(WL_LIST[i][j])):
                    print('WL')
                if(np.isnan(SSC_LIST[i][j])):
                    print('SSC')
                if(np.isnan(ZC_LIST[i][j])):
                    print('ZC')
            entry.append(sensor_list)
            dict_list.append(entry)
    df_car = pd.DataFrame(dict_list)
    df_car.columns = ['label', 'feature']
    return df_car
	
def GROUP_2 (df, WINDOW_SIZE, limiar, v, label):
#    IEMG_LIST   = IEMG_EMG(df, WINDOW_SIZE)
    DASDV_LIST  = DASDV_EMG(df, WINDOW_SIZE)
#    WAMP_LIST   = WAMP_EMG(df, WINDOW_SIZE, limiar)
#    VORDER_LIST = VORDER_EMG(df, WINDOW_SIZE, v)
    AR4_LIST = AR4_EMG(df, WINDOW_SIZE)
    dict_list = []
    for j in range(min([max(df['signal']), 10])):
        if(len(df[df['signal'] == j+1]) >= WINDOW_SIZE):
            entry = [label]
            sensor_list = []
            for i in range(8):
                sensor_list.append(IEMG_LIST[i][j])
                sensor_list.append(DASDV_LIST[i][j])
                sensor_list.append(WAMP_LIST[i][j])
                sensor_list.append(VORDER_LIST[i][j])
                sensor_list.extend(AR4_LIST[i][j])
                if(np.isnan(IEMG_LIST[i][j])):
                    print('IEMG_LIST')
                if(np.isnan(DASDV_LIST[i][j])):
                    print('DASDV_LIST')
                if(np.isnan(WAMP_LIST[i][j])):
                    print('WAMP_LIST')
                if(np.isnan(VORDER_LIST[i][j])):
                    print('VORDER_LIST')
                if(any(np.isnan(AR4_LIST[i][j]))):
                    print('AR4_LIST')
            entry.append(sensor_list)
            dict_list.append(entry)
    df_car = pd.DataFrame(dict_list)
    df_car.columns = ['label', 'feature']
    return df_car