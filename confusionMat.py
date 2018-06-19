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
    name = 'S12'

    imucols = pd.DataFrame(testList)
    data = pd.read_csv(out + name + '.txt', sep="\t")

    kamReal = data.y
    thrshold = 0.7 * np.max(kamReal)
    highdata = data[data.y >= thrshold]
    lowdata = data[data.y < thrshold]

    highdata = highdata.reset_index().iloc[:, 1:]

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

    # predicted = cross_val_predict(model, X, y.values.ravel(), cv=10)
    predicted = model.predict(X)

    from sklearn import metrics

    MSE = metrics.mean_squared_error(y, predicted)
    RMSE = np.sqrt(metrics.mean_squared_error(y, predicted))
    print("MSE:", MSE)
    print("RMSE:", RMSE)

    plt.figure(figsize=(9.06, 9.06))
    p1 = plt.subplot(111)
    # p1.plot(y_demo)
    # p1.plot(demoy_pred)

    std = np.std(predicted)
    print('std:', std)
    # pl.errorbar(y, predicted, yerr=std, fmt="o")
    p1.scatter(y, predicted)

    p1.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    p1.set_title(name, fontsize=20)
    p1.set_xlabel('Measured', fontsize=20)
    p1.set_ylabel('Predicted', fontsize=20)
    plt.savefig('outcome/' + name + ".png")


    output = open('outcome/outcome.txt', 'a')
    output.write('\ntrial:' + name + '\n')
    output.write('kam max:' + str(max(highdata['y'])) + '\n')
    mean = np.mean(highdata['y'])
    output.write('kam mean:' + str(mean) + '\n')
    output.write('MSE:' + str(MSE) + '\n')
    output.write('RMSE:' + str(RMSE) + '\n')
    output.write('RMSE / mean:' + str(RMSE / mean) + '\n')
    output.close()

    plt.show()