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
    name = 'S11'

    imucols = pd.DataFrame(testList)
    data = pd.read_csv(out + name + '.txt', sep="\t")

    kamReal = data.y
    filterIndex = 0.8
    thrshold = filterIndex * np.max(kamReal)

    highdata = data[data.y >= thrshold]
    outlinesIndex = highdata.index
    # highdata = highdata.reset_index().iloc[:, 1:]
    # highdata = highdata.fillna(0)

    lowdata = data[data.y < thrshold]
    lowdata = lowdata.reset_index().iloc[:, 1:]

    traindata = data
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


    predicted = model.predict(X)

    from sklearn import metrics

    MSE = metrics.mean_squared_error(y, predicted)
    RMSE = np.sqrt(metrics.mean_squared_error(y, predicted))
    score = model.score(X, y)

    print("MSE:", MSE)
    print("RMSE:", RMSE)
    print("R2 score:",score)
    print('kam max:' + str(max(data['y'])))
    mean = (np.mean(y))[0]
    print('kam mean:' + str(mean))
    print('RMSE / mean:' + str(RMSE / mean))

    pl.figure()
    p1 = pl.subplot(211)
    p2 = pl.subplot(212)

    p1.plot(np.arange(len(predicted)), y, 'go-', label='true value')
    p1.plot(np.arange(len(predicted)), predicted, 'ro-', label='predict value')
    p1.set_title('prediction')
    p1.legend()

    predicted = pd.DataFrame(predicted)
    predictedHigh = predicted.loc[outlinesIndex]
    realHigh = y.loc[outlinesIndex]
    # predictedHigh = predictedHigh.reset_index().iloc[:, 1:]
    # highdata = highdata.reset_index().iloc[:, 1:]
    p2.scatter(outlinesIndex,realHigh.iloc[:,0], label='true value')
    p2.scatter(outlinesIndex, predictedHigh.iloc[:, 0], label='predict value')

    pl.show()

    predictedHigh = predictedHigh[predictedHigh >= thrshold]
    predictedHigh = predictedHigh.dropna(axis=0, how='all')
    predictedHigh = predictedHigh.reset_index().iloc[:, 1:]

    print(str(100*filterIndex) + '% * max(reald KAM):',thrshold)
    print('number of real data over 0.8 max kam:',(highdata.shape)[0])
    print('number of predicted data over 0.8 max kam:', (predictedHigh.shape)[0])
    print('ratio:', (predictedHigh.shape)[0]/(highdata.shape)[0])

    output = open('outcome/subjectAnalyze.txt', 'a')
    output.write('\nsubject:' + name + '\n')
    output.write("MSE:"+ str(MSE)+ '\n')
    output.write("RMSE:"+ str(RMSE)+ '\n')
    output.write("R2 score:"+ str(score)+ '\n')
    output.write('kam max:' + str(max(data['y']))+ '\n')
    output.write('kam mean:' + str(mean)+ '\n')
    output.write('RMSE / mean:' + str(RMSE / mean)+ '\n')
    output.write(str(100 * filterIndex) + '% * max(reald KAM):'+ str(thrshold)+ '\n')
    output.write('number of real data over 0.8 max kam:' + str((highdata.shape)[0])+ '\n')
    output.write('number of predicted data over 0.8 max kam:'+ str((predictedHigh.shape)[0])+ '\n')
    output.write('ratio:'+str((predictedHigh.shape)[0] / (highdata.shape)[0])+ '\n')
    output.close()
