import pandas as pd
import numpy as np
from dataProcessing import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
import os
from sklearn.externals import joblib


def R2(y_test, y_true):
    return 1 - ((y_test - y_true)**2).sum() / ((y_true - y_true.mean())**2).sum()

def delayedData(imudata,lag=0):
    if(lag == 0):
        return imudata
    # imudata = imudata.iloc[:, 2:]#delete 'time' and 'sync' column
    delayData = pd.DataFrame([])
    delayData = pd.concat([delayData,imudata])
    for i in range(lag):
        delay = i + 1
        temp = pd.DataFrame([])
        temp = pd.concat([temp, imudata])
        temp.columns = imudata.columns + str(delay)

        delayData = delayData.loc[1:].reset_index().iloc[:, 1:]
        lenth = (temp.shape)[0]
        temp = temp.loc[0:lenth-delay-1]
        delayData = pd.concat([temp,delayData],axis=1)
    lenth = (delayData.shape)[0]
    # delayData = delayData.loc[0:lenth-lag-1]
    return delayData

subjectInfo = ['subject','age','mass','height','Lleglen','LkneeWid','Rleglen','LankleWid','RkneeWid','RankleWid','gender_F','gender_M']
tempIotcols = ['LLACx', 'LLACy', 'LLACz', 'LLGYx', 'LLGYy', 'LLGYz'
        , 'LMACx', 'LMACy', 'LMACz', 'LMGYx', 'LMGYy', 'LMGYz'
        , 'RLACx', 'RLACy', 'RLACz', 'RLGYx', 'RLGYy', 'RLGYz'
        , 'RMACx', 'RMACy', 'RMACz', 'RMGYx', 'RMGYy', 'RMGYz','mass']
allcols = ['LLACx','LLACy','LLACz','LLGYx','LLGYy','LLGYz','LMACx','LMACy','LMACz','LMGYx','LMGYy','LMGYz',
           'RLACx','RLACy','RLACz','RLGYx','RLGYy','RLGYz','RMACx','RMACy','RMACz','RMGYx','RMGYy','RMGYz',
           'mass','height','Lleglen','LkneeWid','LankleWid','Rleglen','RkneeWid','y']
infoFile = 'subjectInfo.txt'
out = 'kam2cali/'
iotout = 'kam2allinfo/'
if __name__ == '__main__':
    testList = tempIotcols[0:24]
    lag = 1
    name = 'caliAll-sliced'

    subjects = os.listdir('noraxon')
    subjectFile = getFile(name, subjects)
    imucols = pd.DataFrame(testList)
    data = pd.read_csv(out + name + '.txt', sep="\t")

    kamReal = data.y
    thrshold = 0.8 * np.max(kamReal)
    highdata = data[data.y > thrshold]
    lowdata = data[data.y < thrshold]

    tempdata = highdata[testList]
    delayData = delayedData(tempdata, lag)

    lenth = (highdata.shape)[0]
    useLen = (delayData.shape)[0]
    infoData = highdata[['mass', 'height', 'Lleglen', 'LkneeWid', 'Rleglen', 'LankleWid', 'RkneeWid', 'RankleWid']].loc[
               0:useLen - 1]


    X = pd.concat([delayData, infoData], axis=1)
    y = highdata[['y']].loc[lag:lenth]
    print(X.shape)
    print(y.shape)

    model = joblib.load('sliced-lasso.model')

    y_pred = model.predict(X)  # test
    # demoy_pred = lassoreg.predict(X_demo)
    predicted = cross_val_predict(model, X, y, cv=10)
    print(model.coef_)

    from sklearn import metrics

    MSE = metrics.mean_squared_error(y, y_pred)
    RMSE = np.sqrt(metrics.mean_squared_error(y, y_pred))
    print("MSE:", MSE)
    print("RMSE:", RMSE)

