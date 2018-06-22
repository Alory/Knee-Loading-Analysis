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
    lag = 0
    name = 'caliAll'

    imucols = pd.DataFrame(testList)
    data = pd.read_csv(out + name + '.txt', sep="\t")

    kamReal = data.y
    filterIndex = 0.7
    thrshold = filterIndex * np.max(kamReal)

    highdata = data[data.y >= thrshold]
    highdata = highdata.reset_index().iloc[:, 1:]

    lowdata = data[data.y < thrshold]
    lowdata = lowdata.reset_index().iloc[:, 1:]

    traindata = highdata
    tempdata = traindata[testList]
    delayData = delayedData(tempdata, lag)

    lenth = (traindata.shape)[0]
    useLen = (delayData.shape)[0]
    infoData = traindata[['age','mass','height','Lleglen','LkneeWid','Rleglen','LankleWid','RkneeWid','RankleWid','gender_F','gender_M']].loc[0:useLen-1]


    X = pd.concat([delayData, infoData], axis=1)
    y = traindata[['y']].loc[lag:lenth]
    print(X.shape)
    print(y.shape)

    model = joblib.load('model/' + 'RandomForest-chopped-caliAll.model')
    print(X.columns)
    print(model.oob_prediction_)
    print(model.feature_importances_)
    predicted = model.predict(X)

    from sklearn import metrics

    MSE = metrics.mean_squared_error(y, predicted)
    RMSE = np.sqrt(metrics.mean_squared_error(y, predicted))
    score = model.score(X, y)

    print("MSE:", MSE)
    print("RMSE:", RMSE)
    print("score:",score)
    print('kam max:' + str(max(data['y'])))
    mean = (np.mean(y))[0]
    print('kam mean:' + str(mean))
    print('RMSE / mean:' + str(RMSE / mean))

    pl.figure()
    pl.plot(np.arange(len(predicted)), y, 'go-', label='true value')
    pl.plot(np.arange(len(predicted)), predicted, 'ro-', label='predict value')
    pl.title('score: %f' % score)
    pl.legend()
    pl.show()

    plt.show()

    # thrshold = (0.8 * np.max(y))[0]
    # y[y >= thrshold] = 1
    # y[y < thrshold] = 0
    #
    predicted = pd.DataFrame(predicted)
    predictedHigh = predicted[predicted >= thrshold]
    predictedHigh = predictedHigh.dropna(axis=0, how='all')
    predictedHigh = predictedHigh.reset_index().iloc[:, 1:]
    print(str(100*filterIndex) + '% * max(reald KAM):',thrshold)
    print('number of real data over 0.8 max kam:',(highdata.shape)[0])
    print('number of predicted data over 0.8 max kam:', (predictedHigh.shape)[0])
    #
    # from sklearn.metrics import confusion_matrix
    # tn, fp, fn, tp = confusion_matrix(y, predicted).ravel()
    # size = (y.shape)[0]
    # confMat = np.array([tp, tn, fp, fn, ]) #/ size
    # print(confMat)
