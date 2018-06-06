import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import Lasso

def R2(y_test, y_true):
    return 1 - ((y_test - y_true)**2).sum() / ((y_true - y_true.mean())**2).sum()

tempIotcols = ['LLACx', 'LLACy', 'LLACz', 'LLGYx', 'LLGYy', 'LLGYz'
        , 'LMACx', 'LMACy', 'LMACz', 'LMGYx', 'LMGYy', 'LMGYz'
        , 'RLACx', 'RLACy', 'RLACz', 'RLGYx', 'RLGYy', 'RLGYz'
        , 'RMACx', 'RMACy', 'RMACz', 'RMGYx', 'RMGYy', 'RMGYz','mass']

if __name__ == '__main__':
    data = pd.read_csv('imucom2kam/S16.txt',sep="\t")
    # usecols = list(filter(lambda x:x[2:4]=='GY',tempIotcols))
    # usecols.append('mass')
    usecols = tempIotcols
    X = data[usecols]
    y = data[['y']]

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.4,random_state=152)

    # model = LinearRegression()
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    # predicted = cross_val_predict(model, X, y, cv=10)

    lassoreg = Lasso(alpha=1e-10, normalize=True, max_iter=1e6)
    lassoreg.fit(X_train, y_train)
    y_pred = lassoreg.predict(X_test)
    predicted = cross_val_predict(lassoreg, X, y, cv=10)
    print(lassoreg.coef_)

    a = np.fabs(lassoreg.coef_)
    b = np.sort(a)
    index = []
    sortedpos = []
    for i in range(len(b)):
        pos = np.where(a == b[i])
        pos = pos[0][0]
        index.append(a[pos])
        sortedpos.append(usecols[pos])
    print(index)
    print(sortedpos)

    from sklearn import metrics
    print("MSE:", metrics.mean_squared_error(y_test, y_pred))
    print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    # print("R2:",R2(y_pred,y_test))

    fig, ax = plt.subplots()

    # ax.scatter(X_test,y_test)
    # ax.scatter(X_test,y_pred)

    ax.scatter(y, predicted)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()

