import numpy as np
import scipy.io as sio
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import Lasso

resultDir = "test/"
testdir = "imucom2kam/"
traindir = "kam2mass/"

matFile = "HCTSA-S16_trial3_R.mat"

if __name__ == '__main__':
    triantxt = matFile.split('-')[1][:-4] + '.txt'
    traindata = sio.loadmat(matFile)
    kamdata = pd.read_csv(traindir + triantxt)
    kamy = kamdata.y
    feature = pd.DataFrame(traindata['TS_DataMat'])

    feaNum = (feature.shape)[1]

