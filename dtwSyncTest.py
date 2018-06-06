from functools import reduce
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from dataProcessing import *
import pylab as pl


'''
For testing dynamic time warping method
'''
if __name__ == '__main__':
    iotdatadir = "iot/"
    norxDir = "noraxon/"
    resultDir = "result/"
    testdir = "test/"
    subject = "S10_Fu"
    dir = "S10_0306-subject5"
    trialNum = '7'
    foot = "R"

    rate, imudata = readImuData(norxDir + subject + '\Trial_'+trialNum+'.txt')
    iotFileList = os.listdir(iotdatadir + dir)
    iotdata = readIotData(iotdatadir + dir + "\\" + iotFileList[int(trialNum) - 1], rate)

    iotpos= "RMGYz"
    imupos = iot2imuPos[iotCols.index(iotpos)]
    iotcol = iotdata[iotpos]
    norxcol = flags[iotCols.index(iotpos)] * imudata[imupos]

    normax, normin = peakdet(norxcol, 150)
    iotmax, iotmin = peakdet(iotcol, 150)
    lag = iotmin[0][0]-normin[0][0]
    # distance, path = fastdtw(norxcol, iotcol, dist=euclidean)
    # print(path)
    #
    # iotindex = list(map(lambda i : i[1],path))
    # iotout = []
    # for i in iotindex:
    #     iotout.append(iotcol[i])
    #
    # fabNorx = np.fabs(norxcol)
    # max = norxcol.max()
    # maxPos = np.where(norxcol == max)
    # pos = list(filter(lambda i: i[0] == maxPos[-1], path))
    # print(pos)
    # poslen = int(len(pos)/2)
    # lag = pos[-1][1] - pos[-1][0]
    print(lag)

    pl.figure(figsize=(19.20, 9.06))
    p1 = pl.subplot(311)
    p2 = pl.subplot(312)
    p3 = pl.subplot(313)

    x = np.linspace(0, len(iotcol) - 1, len(iotcol))
    p1.plot(norxcol)
    p1.plot(x-lag,iotcol)
    p2.plot(np.diff(norxcol))
    pl.show()

