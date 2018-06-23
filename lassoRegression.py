import pandas as pd
import numpy as np
from dataProcessing import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LassoCV,Lasso
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
    # testList = list(filter(lambda x: 'GY' in x, tempIotcols))
    testList = tempIotcols[0:24]
    lag = 0
    name = 'allData-L'

    subjects = os.listdir('noraxon')
    subjectFile = getFile(name,subjects)
    # staticData = readStaticData( 'noraxon/' + subjectFile + "/static.txt")
    # staticData = staticData[testList]
    # caliData = getCaliData(staticData)

    imucols = pd.DataFrame(testList)
    data = pd.read_csv(out + name +'.txt',sep="\t")
    tempdata = data[testList]
    # tempdata = tempdata.sub(caliData.iloc[0, :])
    delayData = delayedData(tempdata,lag)

    lenth = (data.shape)[0]
    useLen = (delayData.shape)[0]
    infoData = data[['age','mass','height','Lleglen','LkneeWid','Rleglen','LankleWid','RkneeWid','RankleWid','gender_F','gender_M']].loc[0:useLen-1]

    X = pd.concat([delayData,infoData],axis=1)
    y = data[['y']].loc[lag:lenth]
    print(X.shape)
    print(y.shape)

    seed = 778
    # X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.4,random_state=seed)
    alpha = 1e-7
    iter = 1e7
    tol = 0.000001


    lassoreg = LassoCV(eps=1e-7,cv=10,tol=tol,max_iter=iter,normalize=True).fit(X, y)#normalize=False
    # lassoreg = Lasso(alpha=alpha, normalize=False, max_iter=iter, tol=tol).fit(X, y)
    score = lassoreg.score(X,y)
    alpha = lassoreg.alpha_
    # joblib.dump(lassoreg, 'lasso.model')

    # predicted = cross_val_predict(lassoreg, X, y.values.ravel(), cv=10)
    predicted = lassoreg.predict(X)
    print(lassoreg.coef_)

    m_log_alphas = -np.log10(lassoreg.alphas_)
    plt.figure()
    # ymin, ymax = np.min(y), np.max(y)
    msePath = lassoreg.mse_path_
    rmsePath = np.sqrt(msePath)
    plt.plot(m_log_alphas, rmsePath, ':')
    plt.plot(m_log_alphas, rmsePath.mean(axis=-1), 'k',
             label='Average across the folds', linewidth=2)
    plt.axvline(-np.log10(lassoreg.alpha_), linestyle='--', color='k',
                label='alpha: CV estimate')
    plt.title(name + ' LASSO regression\nRoot mean square error on each fold: coordinate descent ')
    plt.legend()

    plt.xlabel('-log(alpha)')
    plt.ylabel('Root mean square error')
    plt.axis('tight')
    # plt.ylim(ymin, ymax)
    plt.show()

    from sklearn import metrics
    MSE = metrics.mean_squared_error(y, predicted)
    RMSE = np.sqrt(metrics.mean_squared_error(y, predicted))
    print("MSE:", MSE)
    print("RMSE:", RMSE)

    plt.figure(figsize=(9.06, 9.06))
    p1 = plt.subplot(111)

    std = np.std(predicted)
    print('std:',std)
    # pl.errorbar(y, predicted, yerr=std, fmt="o")
    p1.scatter(y, predicted)

    p1.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    p1.set_title(name,fontsize=20)
    p1.set_xlabel('Measured',fontsize=20)
    p1.set_ylabel('Predicted',fontsize=20)
    # plt.savefig('outcome/' + name + '-lag-' + str(lag) + '-alpha-' + str(alpha) + '-iter-' + str(iter) + '-seed-' + str(seed) + ".png")

    cols = X.columns
    a = np.fabs(lassoreg.coef_)
    keys = sorted(range(len(a)), key=lambda k: a[k],reverse=True)
    index = []
    sortedpos = []
    for i in keys:
        index.append((lassoreg.coef_)[i])
        sortedpos.append(cols[i])
    print(index)
    print(sortedpos)

    # output = open('outcome/outcome.txt', 'a')
    # output.write('\ntrial:' + name + '\n')
    # output.write('lag:' + str(lag) + '\n')
    # output.write('score:' + str(score) + '\n')
    # output.write('kam max:' + str(max(data['y'])) + '\n')
    # mean = np.mean(data['y'])
    # output.write('kam mean:' + str(mean) + '\n')
    # output.write('alpha:' + str(alpha) + '\n')
    # output.write('iteration:' + str(lassoreg.n_iter_) + '\n')
    # output.write('tolerance:' + str(tol) + '\n')
    # output.write('MSE:' + str(MSE) + '\n')
    # output.write('RMSE:' + str(RMSE) + '\n')
    # output.write('RMSE / mean:' + str(RMSE/mean) + '\n')
    # output.write('sorted coef:' + str(index) + '\n')
    # output.write('corresponding pos:' + str(sortedpos) + '\n')
    # output.close()



    # plt.show()


