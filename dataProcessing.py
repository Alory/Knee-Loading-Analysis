
"""
@Description : This script is for process the iot, kam, imu data as DataFrame
@Author : Wangchao
@Date : 2018.4.17
"""
import os
import pandas as pd
import numpy as np
from scipy.signal import correlate
import pylab as pl
from scipy import interpolate
from scipy import signal, fftpack
import sys
from numpy import NaN, Inf, arange, isscalar, asarray, array

# imu raw data columns
imuCols = ['time', 'syncOn', 'LLACx', 'LLACy', 'LLACz', 'LLGYx', 'LLGYy', 'LLGYz'
    , 'LMACx', 'LMACy', 'LMACz', 'LMGYx', 'LMGYy', 'LMGYz'
    , 'RLACx', 'RLACy', 'RLACz', 'RLGYx', 'RLGYy', 'RLGYz'
    , 'RMACx', 'RMACy', 'RMACz', 'RMGYx', 'RMGYy', 'RMGYz']
iotCols = ['LLACx', 'LLACy', 'LLACz', 'LLGYx', 'LLGYy', 'LLGYz'
    , 'LMACx', 'LMACy', 'LMACz', 'LMGYx', 'LMGYy', 'LMGYz'
    , 'RLACx', 'RLACy', 'RLACz', 'RLGYx', 'RLGYy', 'RLGYz'
    , 'RMACx', 'RMACy', 'RMACz', 'RMGYx', 'RMGYy', 'RMGYz']
iot2imuCols = {'LLACy': 1, 'LLACx': -1, 'LLACz': 1, 'LLGYy': 1, 'LLGYx': -1, 'LLGYz': 1
        , 'LMACy': -1, 'LMACx': -1, 'LMACz': -1, 'LMGYy': -1, 'LMGYx': -1, 'LMGYz': -1
        , 'RLACy': -1, 'RLACx': -1, 'RLACz': -1, 'RLGYy': -1, 'RLGYx': -1, 'RLGYz': -1
        , 'RMACy': 1, 'RMACx': -1, 'RMACz': 1, 'RMGYy': 1, 'RMGYx': -1, 'RMGYz': 1}
iot2imuPos = list(iot2imuCols.keys())
flags = list(iot2imuCols.values())
hongkongG = 978.5

std = np.array([hongkongG, 0, 0])
std = np.mat(std)#.transpose()

subjectMass = {'S8':67.7,'S10':59.5,'S11':66.3,'S12':67.6,'S13':45.4,'S15':53.6,'S16':88.7,'S17':61.5,
        'S18':54.9,'S21':85.7,'S22':64.1,'S23':85.8}

def interpolateData(data):
    m = len(data)
    n = len(data[0])
    x = np.linspace(1, m, m)
    y = np.linspace(1, n, n)
    xnew = np.linspace(0, m, 2 * m)
    f = interpolate.interp2d(y, x, data, kind='linear')
    newData = f(y, xnew)
    return newData


'''
read kam data and then interpolate data(up sampling).
use linear interpolating.
'''
def interpolateDataTest(file, num):
    data = np.loadtxt(file, delimiter="\t", skiprows=1)
    m = len(data)
    n = len(data[0])
    x = np.linspace(1, m, m)
    y = np.linspace(1, n, n)
    xnew = np.linspace(0, m, num)
    f = interpolate.interp2d(y, x, data, kind='linear')
    new = f(y, xnew)
    df = pd.DataFrame(new, columns=['x', 'y', 'z'])
    pl.plot(data[:, 1], label='data')
    pl.plot(new[:, 1], label='new')
    pl.legend(loc="lower right")
    pl.show()


'''
read iot data and process the dummy elements
return dataframe
'''
def readIotData(filename, imuRate=100):
    if (imuRate == 100):
        data = pd.read_table(filename)  # "test.txt"
    else:
        rawData = np.genfromtxt(filename, delimiter="\t", skip_header=1)
        npdata = interpolateData(rawData)
        data = pd.DataFrame(npdata)
        data.columns = iotCols
    for pos in iotCols:
        data[pos] = signal.savgol_filter(data[pos], 7, 3)
    return data


'''
read IMU raw data(dataframe)
'''
def readImuData(filename):
    info = pd.read_table(filename, nrows=3)  # read only first 4 rows
    valueCols = [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16,
                 20, 21, 22, 23, 24, 25, 29, 30, 31, 32, 33, 34]
    data = pd.read_table(filename, skiprows=4, sep="\t", usecols=valueCols)  # skip first 4 rows
    imuRate = int(info.iat[1, 0])  # get imu rate
    data.columns = imuCols
    for pos in iotCols:
        data[pos] = signal.savgol_filter(data[pos], 7, 3)
    return imuRate, data

'''
read static data(dataframe)
'''
def readStaticData(filename):
    valueCols = [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16,
                 20, 21, 22, 23, 24, 25, 29, 30, 31, 32, 33, 34]
    data = pd.read_table(filename, skiprows=4, sep="\t", usecols=valueCols)  # skip first 4 rows
    data.columns = imuCols
    for pos in iotCols:
        data[pos] = signal.savgol_filter(data[pos], 7, 3)
    return data

'''
get the sync point of iot data and imu data
return lag
'''
def getIotImuLag(iotV,norV,flag=1):
    lenIot = int(len(iotV)*3/4)
    lenNorx = len(norV)
    realLen = len(iotV)
    maxLen = lenIot if lenIot>lenNorx else lenNorx

    iotV = flag*iotV.reset_index().iloc[:,1]
    norV = norV.reset_index().iloc[:,1]
    mean = iotV[1:100].mean()
    iotTemp = iotV[0:lenIot] - mean

    iotTemp = np.concatenate((iotTemp, np.zeros((maxLen - lenIot), dtype=int)))
    norV = np.concatenate((norV, np.zeros((maxLen - lenNorx), dtype=int)))
    cor = correlate(norV, iotTemp,method="fft")
    lag = np.argmax(cor) - cor.shape[0] / 2
    print(lag)


    p1 = pl.subplot(311)
    p2 = pl.subplot(312)
    p3 = pl.subplot(313)
    p4 = pl.subplot(313)

    # lag = 70
    x = np.linspace(0, realLen - 1, realLen)
    p1.plot(norV)
    p2.plot(iotV)
    p1.plot(x+lag, iotV, color='red')  # x+lag,iotV
    # p1.plot(shiftSig, color='red')
    p4.plot(cor)
    pl.show()
    return lag,cor


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


def getIotImuLagFFT(iotV,norV,flag=1):
    iotV = flag * iotV.reset_index().iloc[:,1]
    norV = norV.reset_index().iloc[:,1]

    # iotV = signal.medfilt(iotV,5)
    # norV = signal.medfilt(norV,5)
    # cutOff = 70
    # fs = 200
    # order = 6
    # iotV = butter_lowpass_filter(iotV, cutOff, fs, order)
    # norV = butter_lowpass_filter(norV, cutOff, fs, order)

    # iotV = signal.savgol_filter(iotV,7,3)
    # norV = signal.savgol_filter(norV,7,3)

    mean = iotV[1:100].mean()

    iotStart = int(len(iotV) * 1 / 3)
    iotEnd = int(len(iotV))
    norStart = int(len(norV) * 1 / 4)
    norEnd = int(len(norV) * 4 / 5)


    iotLen = iotEnd - iotStart
    norLen = norEnd - norStart
    iotTemp = iotV[iotStart:iotEnd].reset_index().iloc[:,1] - mean
    norTemp = norV[norStart:norEnd].reset_index().iloc[:,1]
    # iotTemp = iotV[iotStart:iotEnd] - mean
    # norTemp = norV[norStart:norEnd]
    print("shape:",iotLen,norLen)

    norData = fftpack.fft(norTemp,iotLen+norLen)
    iotData = fftpack.fft(iotTemp,iotLen+norLen)

    norDatar = -norData.conjugate()
    iotDatar = iotData.conjugate()

    cor = np.abs(fftpack.ifft(norDatar * iotData))
    lag = np.argmax(cor)

    lenIot = len(iotV)
    lenNorx = len(norV)
    pl.figure(figsize=(19.20, 9.06))
    p1 = pl.subplot(311)
    p2 = pl.subplot(312)
    p3 = pl.subplot(313)
    p4 = pl.subplot(313)

    if(lag > iotLen):
        lag = lag - (iotLen+norLen)
    lag = lag + (iotStart-norStart)
    print(lag)
    # lag = 25
    x = np.linspace(0, lenIot - 1, lenIot)
    p1.plot(norV)
    p2.plot(iotV)
    p2.plot(iotTemp)
    p1.plot(x-lag, iotV, color='red')  # x+lag,iotV
    p3.plot(cor)

    pl.show()
    return lag

# def lagTest(iotV,norV,flag=1):
def get_max_correlation(iotV, norV, flag=1):
    iotV = flag*iotV
    z = signal.fftconvolve(norV, iotV[::-1])
    lags = np.arange(z.size) - (iotV.size - 1)
    lag = lags[np.argmax(np.abs(z))]

    p1 = pl.subplot(311)
    p2 = pl.subplot(312)
    p3 = pl.subplot(313)
    lenIot = len(iotV)
    x = np.linspace(0, lenIot - 1, lenIot)
    p1.plot(norV)
    p2.plot(iotV)
    p1.plot(x - lag, iotV, color='red')
    pl.show()
    return lag

'''
read kam file, and then transform to data frame
return dataframe
'''
def readKam(filename,imurate = 100):
    #data = np.genfromtxt(filename, delimiter="\t", skip_header=3,missing_values=np.nan, usecols=(1, 2, 3, 4))#, dtype=bytes).astype(str)
    #df = pd.DataFrame(data, columns=['Frames', 'x', 'y', 'z'],index=data[:,0])#'Frame',

    if(imurate == 100):
        data = pd.read_csv(filename, sep="\t", skiprows=3, usecols=[1, 2, 3, 4], index_col=0)
        data.columns = ['x', 'y', 'z']
    else:#200Hz
        print("rate:",imurate)
        npdata = np.genfromtxt(filename, delimiter="\t", skip_header=3, usecols=(2, 3, 4),filling_values=0.00)
        tmpdata = interpolateData(npdata)
        data = pd.DataFrame(tmpdata)
        data.columns = ['x', 'y', 'z']#'Frame',
        data.index = data.index + 2
        # data.to_csv("out.txt")
    return data


'''
read result file containing the usable frame range, and process dummy variable L/R
return dataframe
cols:["Trial No.","L/R_L","L/R_R","On","Off"]
'''
def getFrame(resultFile):
    result = pd.read_csv(resultFile,delimiter=",")
    df = pd.get_dummies(result)
    return df

'''
get two sequences lag
'''
def getSeqLag(norV, iotV, flag=1):
    # iotV = np.concatenate((iotV, np.zeros((norV.shape[0] - iotV.shape[0]), dtype=int)))
    #
    # iotV = np.concatenate((iotV, np.zeros((1000 - iotV.shape[0]), dtype=int)))
    # norV = np.concatenate((norV, np.zeros((1000 - norV.shape[0]), dtype=int)))
    # norV = norV[0:iotV.shape[0]]
    cor = correlate(norV, iotV, method='fft')
    if (flag == 1):
        lag = np.argmax(cor) - cor.shape[0] / 2
    else:
        lag = np.argmin(cor) - cor.shape[0] / 2
    pl.plot(cor)
    pl.show()
    return lag

def getSeqLagFFT(iotV,norV,flag):
    iotV = flag * iotV.reset_index().iloc[:, 1]
    norV = norV.reset_index().iloc[:, 1]
    mean = iotV[1:100].mean()

    iotStart = int(len(iotV) * 1 / 3)
    iotEnd = int(len(iotV))
    norStart = int(len(norV) * 1 / 4)
    norEnd = int(len(norV) * 4 / 5)

    iotLen = iotEnd - iotStart
    norLen = norEnd - norStart
    iotTemp = iotV[iotStart:iotEnd].reset_index().iloc[:, 1] - mean
    norTemp = norV[norStart:norEnd].reset_index().iloc[:, 1]

    norData = fftpack.fft(norTemp, iotLen + norLen)
    iotData = fftpack.fft(iotTemp, iotLen + norLen)

    norDatar = norData.conjugate()
    # iotDatar = iotData.conjugate()

    cor = np.abs(fftpack.ifft(norDatar * iotData))
    lag = np.argmax(cor)
    if(lag > iotLen):
        lag = lag - (iotLen+norLen)
    lag = lag + (iotStart - norStart)
    return -lag

'''
return an dictionary of lags
{LL:,LM:,RM:,RL:}
'''
def getDataLag(norData,iotData):
    pos = ["LLGYz", "LMGYz", "RMGYz", "RLGYz"]
    # npos = ["LLACx", "LMACx", "RMACx", "RLACx"]
    # mypos = ["LLACy", "LMACy", "RMACy", "RLACy"]
    flags = [1,-1,1,-1]
    # flags = [-1, -1, -1, -1]
    lags = {"LL":None,"LM":None,"RM":None,"RL":None}
    for ipos in range(len(pos)):
        norv = norData[pos[ipos]]
        iotv = iotData[pos[ipos]]
        # lags.append(getSeqLag(norv,iotv,flags[ipos]))
        lags[pos[ipos][:2]] = int(getSeqLag(norv,iotv,flags[ipos]))
    return lags

#get lag of ["LLGYz", "LMGYz", "RMGYz", "RLGYz"]
def getDataLagFFT4(norData,iotData):
    pos = ["LLGYz", "LMGYz", "RMGYz", "RLGYz"]
    # npos = ["LLACx", "LMACx", "RMACx", "RLACx"]
    # mypos = ["LLACy", "LMACy", "RMACy", "RLACy"]
    flags = [1,-1,1,-1]
    # flags = [-1, -1, -1, -1]
    lags = {"LL":None,"LM":None,"RM":None,"RL":None}
    for ipos in range(len(pos)):
        norv = norData[pos[ipos]]
        iotv = iotData[pos[ipos]]
        # lags.append(getSeqLag(norv,iotv,flags[ipos]))
        lags[pos[ipos][:2]] = int(getSeqLagFFT(norv,iotv,flags[ipos]))
    return lags

#get lag of every axis
def getDataLagFFT16(norData,iotData):
    lags = {}
    for ipos in range(len(iotCols)):
        iotPos = iotCols[ipos]
        norxPos = iot2imuPos[ipos]
        norv = norData[norxPos]
        iotv = iotData[iotPos]
        lags[iotCols[ipos]] = int(getSeqLagFFT(iotv,norv,flags[ipos]))

    count = 0
    subLags = []
    templags = []
    for pos in lags:
        count = count + 1
        templags.append(lags[pos])

        if(count % 3 == 0):
            median = get_median(templags)
            subLags = subLags + [median] * 3
            templags = []
            count = 0
    for ipos in range(len(subLags)):
        lag = subLags[ipos]
        lags[iotCols[ipos]] = lag



    return lags

#slice the usable iot data for training
def getIot2Kam(lags,iotData,kamData,imuOn,imuOff,shift=0):
    usableLen = imuOff - imuOn
    #取出最大值
    # iot2kam = pd.DataFrame([])
    iot2kam = pd.DataFrame([max(kamData.y)]*(usableLen + 2*shift))
    iot2kam.columns = ['KAMy']
    for ipos in range(len(iotCols)):
        iotPos = iotCols[ipos]
        norxPos = iot2imuPos[ipos]
        flag = flags[ipos]

        iotOnLL = imuOn - lags[iotPos] - shift
        iotOffLL = imuOff - lags[iotPos] + shift
        iposData = iotData[iotPos][iotOnLL:iotOffLL].reset_index().iloc[:,1]
        # iot2kam = iot2kam.join(iotData, axis=1)
        iot2kam = pd.concat([iposData,iot2kam], axis=1)
    return iot2kam

#slice the noraxon data for training
def getImu2Kam(imuData,kamData,imuOn,imuOff,mass):
    usableLen = imuOff - imuOn
    subjectMass = pd.DataFrame([mass]*usableLen)
    iot2kam = pd.DataFrame([])
    kamy = kamData.y
    kamy = kamy.reset_index().iloc[:,1]
    dataCols = imuCols[2:]
    imuData = imuData[imuOn:imuOff]
    for ipos in range(len(dataCols)):
        imuPos = dataCols[ipos]

        iposData = imuData[imuPos].reset_index().iloc[:,1]
        iot2kam = pd.concat([iot2kam,iposData], axis=1)
    iot2kam = pd.concat([iot2kam,kamy,subjectMass], axis=1)
    return iot2kam

#traverse kam directory and return the list of kam files
def filterKamList(resultFile):
    kamList = []
    resultList = os.listdir(resultFile)
    for item in resultList:
        if(item[-4:] == ".txt"):
            info = item[:-4].split("_")
            trialNum = info[1]
            foot = info[2]
            tail = info[3]
            kamPos = {"file":item,"trialNum":trialNum,"foot":foot,"tail":tail}
            kamList.append(kamPos)
    return kamList

#find particular data file in a file list
def getFile(name,fileList):
    for file in fileList:
        if (name in file):
            return file

def rfft_xcorr(x, y):
    M = len(x) + len(y) - 1
    N = 2 ** int(np.ceil(np.log2(M)))
    X = np.fft.rfft(x, N)
    Y = np.fft.rfft(y, N)
    cxy = np.fft.irfft(X * np.conj(Y))
    cxy = np.hstack((cxy[:len(x)], cxy[N-len(y)+1:]))
    return cxy

def match(iotV, norV, flag):
    iotV = flag * iotV.reset_index().iloc[:, 1]
    norV = norV.reset_index().iloc[:, 1]
    mean = iotV[1:100].mean()

    iotStart = int(len(iotV) * 2 / 5)
    iotEnd = int(len(iotV))
    norStart = int(len(norV) * 1 / 4)
    norEnd = int(len(norV) * 4 / 5)

    iotLen = iotEnd - iotStart
    norLen = norEnd - norStart
    iotTemp = iotV[iotStart:iotEnd].reset_index().iloc[:, 1] - mean
    norTemp = norV[norStart:norEnd].reset_index().iloc[:, 1]

    # iotV = flag * iotV
    cor = rfft_xcorr(iotTemp, norTemp)
    index = np.argmax(cor)
    if index < iotLen:
        lag = index
    else: # negative lag
        lag = index - len(cor)
    lag = lag + (iotStart - norStart)

    lenIot = len(iotV)
    lenNorx = len(norV)
    p1 = pl.subplot(311)
    p2 = pl.subplot(312)
    p3 = pl.subplot(313)

    x = np.linspace(0, lenIot - 1, lenIot)
    p1.plot(norV)
    p2.plot(iotV)
    p1.plot(x - lag, iotV, color='red')  # x+lag,iotV
    p3.plot(cor)

    pl.show()
    print(lag)
    return lag

#get median
def get_median(data):
    data.sort()
    half = len(data)// 2
    median = int((data[half] + data[~half]) / 2)
    return median

def peakdet(v, delta, x=None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html

    Returns two arrays

    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.

    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.

    """
    maxtab = []
    mintab = []

    if x is None:
        x = arange(len(v))

    v = asarray(v)

    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')

    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')

    if delta <= 0:
        sys.exit('Input argument delta must be positive')

    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN

    lookformax = True

    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx - delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn + delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)

def getSeqLagbyPeak(norV,iotV,flag):
    norV = flag * norV
    iotcol = iotV
    norxcol = norV

    normax, normin = peakdet(norxcol, 150)
    iotmax, iotmin = peakdet(iotcol, 150)
    lag = normin[0][0] -iotmin[0][0]

    return lag


def peakLag(norData,iotData):
    pos = ["LLGYz", "LMGYz", "RMGYz", "RLGYz"]
    # flags = [1, -1, 1, -1]
    gylags = {"LL": None, "LM": None, "RM": None, "RL": None}
    for ipos in pos:
        norv = norData[ipos]
        imupos = iot2imuPos[iotCols.index(ipos)]
        iotv = iotData[imupos]
        gylags[ipos[:2]] = int(getSeqLagbyPeak(norv, iotv, flags[iotCols.index(ipos)]))

    lags = {}
    for ipos in range(len(iotCols)):
        lags[iotCols[ipos]] = gylags[iotCols[ipos][0:2]]
    return lags

def delayedData(imudata,lag):
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
        temp = temp.loc[0:lenth-i]
        delayData = pd.concat([delayData,temp],axis=1)
    return delayData

def getCaliData(staticData):
    # staticData.loc['std'] = staticData.apply(lambda x: x.std(ddof=0))
    staticData.loc['std'] = staticData.apply(lambda x: x.mean())
    caliData = staticData.iloc[-1:]
    # acx = ['LMACx', 'LLACx', 'RLACx', 'RMACx']
    # caliData[acx] = caliData[acx] - hongkongG

    acAxis = {'llac': ['LLACx', 'LLACy', 'LLACz'],
              'lmac': ['LMACx', 'LMACy', 'LMACz'],
              'rlac': ['RLACx', 'RLACy', 'RLACz'],
              'rmac': ['RMACx', 'RMACy', 'RMACz']}
    rotMat = {}
    for axis in acAxis:
        data = caliData[acAxis[axis]]
        data = np.mat(data)
        # data = data.transpose()
        R,t = rigid_transform(data)
        rotMat[axis] = [R,t]


    return caliData,rotMat


# B = R*A + t
# get the rotation matrix and translation vector r and t
def rigid_transform(A):
    print(A)
    B = std
    assert len(A) == len(B)

    N = A.shape[0];  # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = np.transpose(AA) * BB

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T * U.T

    t = -R * centroid_A.T + centroid_B.T

    return R, t

# according to 3 axis accelerometer data to get the rotate matrix
def getRotMat(data):
    data = data.values
    k = np.dot(data.transpose(), data)
    k = k[0][0]
    A = (np.dot(std, data)) / k

    return A

# input trial data and the rotate matrix dictionary, output the calibrated data
def caliData(data,rotMat):
    acAxis = {'llac': ['LLACx', 'LLACy', 'LLACz'],
              'lmac': ['LMACx', 'LMACy', 'LMACz'],
              'rlac': ['RLACx', 'RLACy', 'RLACz'],
              'rmac': ['RMACx', 'RMACy', 'RMACz']}

    for indexs in data.index:
        for axis in acAxis:
            pos = acAxis[axis]
            R = rotMat[axis][0]
            t = rotMat[axis][1]
            m = data.loc[indexs, pos]
            out = R*m.T + t
            out.shape = [1, 3]
            out = np.array(out)
            data.loc[indexs, pos] = out[0]

    return data











