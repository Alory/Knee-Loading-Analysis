import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import Lasso

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
out = 'kam2allinfo/'
if __name__ == '__main__':
    # testList = list(filter(lambda x: 'GY' in x, tempIotcols))
    testList = tempIotcols[0:24]
    lag = 2
    name = 'S22'

    imucols = pd.DataFrame(testList)

    data = pd.read_csv('kam2allinfo/'+name +'.txt',sep="\t")
    tempdata = data[testList]
    delayData = delayedData(tempdata,lag)

    lenth = (data.shape)[0]
    useLen = (delayData.shape)[0]
    infoData = data[['mass','height','Lleglen','LkneeWid','Rleglen','LankleWid','RkneeWid','RankleWid']].loc[0:useLen-1]

#=======
    # demo = pd.read_csv('kam2allinfo/demo.txt', sep="\t")
    # tempdata = demo[testList]
    # demodelayData = delayedData(tempdata, lag)
    #
    # demolenth = (demo.shape)[0]
    # demouseLen = (demodelayData.shape)[0]
    # demoinfoData = demo[['mass', 'height', 'Lleglen', 'LkneeWid', 'Rleglen', 'LankleWid', 'RkneeWid', 'RankleWid']].loc[
    #            0:demouseLen - 1]
    # X_demo = pd.concat([demodelayData,demoinfoData],axis=1)
    # y_demo = demo[['y']].loc[lag:demolenth]
# =======

    X = pd.concat([delayData,infoData],axis=1)
    y = data[['y']].loc[lag:lenth]
    print(X.shape)
    print(y.shape)

    seed = 16
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.4,random_state=seed)

    # model = LinearRegression()
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    # predicted = cross_val_predict(model, X, y, cv=10)

    alpha = 1e-7
    iter = 1e6
    lassoreg = Lasso(alpha=alpha, normalize=True, max_iter=iter)
    lassoreg.fit(X_train, y_train)
    y_pred = lassoreg.predict(X_test)#test
    demoy_pred = lassoreg.predict(X_demo)
    predicted = cross_val_predict(lassoreg, X, y, cv=10)
    print(lassoreg.coef_)

    from sklearn import metrics

    MSE = metrics.mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    print("MSE:", MSE)
    print("RMSE:", RMSE)
    # print("R2:",R2(y_pred,y_test))

    plt.figure(figsize=(19.20, 9.06))
    p1 = plt.subplot(111)
    # p1.plot(y_demo)
    # p1.plot(demoy_pred)

    p1.scatter(y, predicted)
    p1.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    p1.set_xlabel('Measured')
    p1.set_ylabel('Predicted')
    plt.savefig('outcome/' + name + '-lag-' + 'alpha-' + str(alpha) + '-iter-' + str(iter) + str(lag) + '-seed-' + str(seed) + ".png")

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

    output = open('outcome/outcome.txt', 'a')
    output.write('\ntrial:' + name + '\n')
    output.write('lag:' + str(lag) + '\n')
    output.write('seed:' + str(seed) + '\n')
    output.write('kam max:' + str(max(data['y'])) + '\n')
    output.write('alpha:' + str(alpha) + '\n')
    output.write('max iter num:' + str(iter) + '\n')
    output.write('MSE:' + str(MSE) + '\n')
    output.write('RMSE:' + str(RMSE) + '\n')
    output.write('sorted coef:' + str(index) + '\n')
    output.write('corresponding pos:' + str(sortedpos) + '\n')
    output.close()



    plt.show()


