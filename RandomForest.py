import random

import pandas as pd
import numpy as np
from dataProcessing import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
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

def plorPredict(clf,x_train,y_train,x_test,y_test):
    clf.fit(x_train,y_train)
    score = clf.score(x_test, y_test)
    result = clf.predict(x_test)
    plt.figure()
    plt.plot(np.arange(len(result)), y_test,'go-',label='true value')
    plt.plot(np.arange(len(result)),result,'ro-',label='predict value')
    plt.title('score: %f'%score)
    plt.legend()
    plt.show()

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
    # testList = list(filter(lambda x: 'AC' in x, tempIotcols))
    testList = tempIotcols[0:24]
    lag = 0
    name = 'allData-L'

    subjects = os.listdir('noraxon')
    subjectFile = getFile(name,subjects)

    imucols = pd.DataFrame(testList)
    data = pd.read_csv(out + name +'.txt',sep="\t")

    # thrshold = 0.01
    # data = data[data.y >= thrshold]
    # data = data.reset_index().iloc[:, 1:]

    tempdata = data[testList]
    delayData = delayedData(tempdata,lag)
    lenth = (data.shape)[0]
    useLen = (delayData.shape)[0]
    infoData = data[['age', 'mass', 'height', 'Lleglen', 'LkneeWid', 'Rleglen', 'LankleWid', 'RkneeWid', 'RankleWid',
                     'gender_F', 'gender_M']].loc[0:useLen - 1]

    X = pd.concat([delayData, infoData], axis=1)
    y = data[['y']].loc[lag:lenth]
    print(X.shape)
    print(y.shape)

    seed = random.randint(1, 200)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=seed)

    from sklearn import ensemble

    method = 'RandomForest'

    model = ensemble.RandomForestRegressor(n_estimators=50,oob_score=True,bootstrap=True)#,max_features=20
    model.fit(X_train, y_train)


    joblib.dump(model, 'model/' + method + '-' + name + '.model')

    trainScore = model.score(X_train, y_train)
    testScore = model.score(X_test, y_test)
    predicted = model.predict(X_test)

    from sklearn import metrics
    MSE = metrics.mean_squared_error(y_test, predicted)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, predicted))
    print("MSE:", MSE)
    print("RMSE:", RMSE)

    # pl.figure()
    # pl.plot(np.arange(len(predicted)), y_test, 'go-', label='true value')
    # pl.plot(np.arange(len(predicted)), predicted, 'ro-', label='predict value')
    # pl.title('score: %f' % score)
    # pl.legend()
    # pl.show()

    plt.figure(figsize=(9.06, 9.06))
    p1 = plt.subplot(111)
    p1.scatter(y_test, predicted)
    p1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    p1.set_title(name, fontsize=20)
    p1.set_xlabel('Measured', fontsize=20)
    p1.set_ylabel('Predicted', fontsize=20)
    plt.savefig('outcome/' + name + '-lag-' + str(lag) + '-method-' + method + str(seed) + ".png")

    output = open('outcome/outcome.txt', 'a')
    output.write('\ntrial:' + name + '\n')
    output.write('data size:' + str(data.shape) + '\n')
    output.write('method:' + method + '\n')
    output.write('training dataset R2 score:' + str(trainScore) + '\n')
    output.write('test dataset R2 score:' + str(testScore) + '\n')
    output.write('OOB score:' + str(model.oob_score_) + '\n')
    output.write('kam max:' + str(max(data['y'])) + '\n')
    mean = np.mean(y_test['y'])
    output.write('kam mean:' + str(mean) + '\n')
    output.write('MSE:' + str(MSE) + '\n')
    output.write('RMSE:' + str(RMSE) + '\n')
    output.write('RMSE / mean:' + str(RMSE / mean) + '\n')
    output.close()

    plt.show()





























