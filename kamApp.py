#!/usr/bin/env python
# Implementation of algorithm from http://stackoverflow.com/a/22640362/6029703
from dataProcessing import *

def thresholding_algo(y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y) - 1):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag):i])
            stdFilter[i] = np.std(filteredY[(i-lag):i])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag):i])
            stdFilter[i] = np.std(filteredY[(i-lag):i])

    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter))
out = 'kam2cali/'
iotout = 'kam2allinfo/'
iotdatadir = "iot/"
norxDir = "noraxon/"
resultDir = "test/"
testdir = "test/"
subject = "S32_Leng"
dir = "S32_0430-subject1"
trialNum = '2'

if __name__ == '__main__':
    # Data
    rate, imudata = readImuData(norxDir + subject + '\Trial_' + trialNum + '.txt')
    y = imudata['LLGYz'][300:]
    y = np.array(y.reset_index().iloc[:, 1:])

    # Settings: lag = 30, threshold = 5, influence = 0
    lag = 10
    threshold = 4
    influence = 0

    # Run algo with settings from above
    result = thresholding_algo(y, lag=lag, threshold=threshold, influence=influence)

    # Plot result
    p1 = pl.subplot(211)
    p2 = pl.subplot(212)
    p1.plot(np.arange(1, len(y) + 1), y)

    p1.plot(np.arange(1, len(y) + 1),
               result["avgFilter"], color="cyan", lw=2)

    p1.plot(np.arange(1, len(y) + 1),
               result["avgFilter"] + threshold * result["stdFilter"], color="green", lw=2)

    p1.plot(np.arange(1, len(y) + 1),
               result["avgFilter"] - threshold * result["stdFilter"], color="green", lw=2)

    p2.step(np.arange(1, len(y) + 1), result["signals"], color="red", lw=2)
    pl.ylim(-1.5, 1.5)
    pl.show()