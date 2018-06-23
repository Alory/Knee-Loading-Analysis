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

modelL = joblib.load('model/' + 'RandomForest-allData-L.model')
modelR = joblib.load('model/' + 'RandomForest-allData-R.model')

if __name__ == '__main__':
    testList = tempIotcols[0:24]
    lag = 0
    # names = ['S7','S8','S10','S11','S12','S16','S21','S22','S23','S26','S28','S29','S30','S31','S32','S33','S35','S37','S38','S39','S40','S41']
    names = ['S41']
    feet = ['L','R']

    for name in names:
        for foot in feet:
            subname = name + '-' + foot
            imucols = pd.DataFrame(testList)
            data = pd.read_csv(out + subname + '.txt', sep="\t")

            # highdata = highdata.reset_index().iloc[:, 1:]
            # highdata = highdata.fillna(0)

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

            if(foot == 'L'):
                model = modelL
            if(foot == 'R'):
                model = modelR

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

            pl.figure(figsize=(10,7))
            p1 = pl.subplot(211)
            p2 = pl.subplot(212)
            # p3 = pl.subplot(313)

            # p3.scatter(peakIndex, peakValues, color='blue', marker='*')
            p1.plot(np.arange(len(predicted)), y, 'go-', label='true value')
            p1.plot(np.arange(len(predicted)), predicted, 'ro-', label='predict value')

            p1.set_title(subname + ' prediction\n' + 'R2:' + str(score))
            p1.set_xlabel('index')
            p1.set_ylabel('KAM in Y axis')
            p1.legend()

            filterIndex = 0.8
            maxtab, mintab = peakdet(y, 0.01)
            peakIndex = maxtab[:, 0]
            peakValues = maxtab[:, 1]
            thrshold = filterIndex * np.mean(peakValues)

            highdata = data[data.y >= thrshold]
            lowdata = data[data.y < thrshold]
            outlinesIndex = highdata.index
            lowdataIndex = lowdata.index

            predicted = pd.DataFrame(predicted)
            predictedHigh = predicted.loc[outlinesIndex]
            realHigh = y.loc[outlinesIndex]

            allNum = (y.shape)[0]
            predictedHighNum = (predictedHigh.shape)[0]#predicted value over 0.8max
            predictedLowNum = allNum - predictedHighNum
            realHighNum = (realHigh.shape)[0]
            realLowNum = allNum - realHighNum

            p2.set_title('data over 80% of max KAM')
            p2.scatter(outlinesIndex,realHigh.iloc[:,0],c='g',marker='o')
            p2.scatter(outlinesIndex, predictedHigh.iloc[:, 0],c='r',marker='o')

            # predictedHigh = predictedHigh[predictedHigh >= thrshold]
            # predictedHigh = predictedHigh.dropna(axis=0, how='all')
            # predictedHigh = predictedHigh.reset_index().iloc[:, 1:]
            # print(str(100*filterIndex) + '% * max(reald KAM):',thrshold)
            # print('number of real data over 0.8 max kam:',(highdata.shape)[0])
            # print('number of predicted data over 0.8 max kam:', (predictedHigh.shape)[0])
            # print('ratio:', (predictedHigh.shape)[0]/(highdata.shape)[0])

            y[y < thrshold] = 0
            y[y >= thrshold] = 1
            predicted[predicted < thrshold] = 0
            predicted[predicted >= thrshold] = 1

            from sklearn.metrics import confusion_matrix
            mat = confusion_matrix(y, predicted)
            # tn, fp, fn, tp = mat.ravel()
            mat = np.array(mat,dtype=float)
            mat[0,0] = mat[0,0] / realLowNum
            mat[1,1] = mat[1,1] / realHighNum
            mat[0,1] = mat[0,1] / realLowNum
            mat[1,0] = mat[1,0] / realHighNum

            pl.savefig('outcome/' + subname + '-prediction.png')
            fig = print_confusion_matrix(mat,['< 0.8*max','>= 0.8*max'], fontsize=8,title=subname)

            pl.savefig('outcome/' + subname + 'confusionMat.png')
            pl.show()

            result = [score,RMSE,mean, RMSE / mean, mat[1,1], mat[0, 0], mat[0, 1], mat[1, 0]]
            for i in range(len(result)):
                result[i] = '%.4f' % result[i]
            result = '\t'.join(result)
            # output = open('outcome/subjectAnalyze.txt', 'a')
            # output.write('\nsubject:' + subname + '\n')
            # output.write("MSE:"+ str(MSE)+ '\n')
            # output.write("RMSE:"+ str(RMSE)+ '\n')
            # output.write("R2 score:"+ str(score)+ '\n')
            # output.write('kam max:' + str(max(data['y']))+ '\n')
            # output.write('kam mean:' + str(mean)+ '\n')
            # output.write('RMSE / mean:' + str(RMSE / mean)+ '\n')
            # output.write('TN:' + str(mat[0, 0])+ '\n')
            # output.write('TP:' + str(mat[1, 1]) + '\n')
            # output.write('FP:' + str(mat[0, 1]) + '\n')
            # output.write('FN:' + str(mat[1, 0]) + '\n')

            output = open('outcome/testresult.txt', 'a')
            output.write(subname + '\t' + result + '\n')
            output.close()
