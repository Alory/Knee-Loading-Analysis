import pandas as pd
import numpy as np
import pylab as pl
from scipy import interpolate
from dataProcessing import *
import os

'''
imuCols = ['time', 'syncOn', 'LLACx', 'LLACy', 'LLACz', 'LLGYx', 'LLGYy', 'LLGYz'
    , 'LMACx', 'LMACy', 'LMACz', 'LMGYx', 'LMGYy', 'LMGYz'
    , 'RLACx', 'RLACy', 'RLACz', 'RLGYx', 'RLGYy', 'RLGYz'
    , 'RMACx', 'RMACy', 'RMACz', 'RMGYx', 'RMGYy', 'RMGYz']
'''

if __name__ == '__main__':
    iotdatadir = "iot/"
    norxDir = "noraxon/"
    resultDir = "test/"
    testdir = "test/"
    subject = "S8_Yeung"
    dir = "S8_0306-subject3"
    trialNum = '1'
    foot = "R"

    rate, imudata = readImuData(norxDir + subject + '\Trial_'+trialNum+'.txt')
    iotFileList = os.listdir(iotdatadir + dir)
    # resultList = filterKamList(resultDir + subject)

    iotdata = readIotData(iotdatadir + dir + "\\" + iotFileList[int(trialNum)-1], rate)
    frameRange = getFrame(resultDir + subject + "\\" + subject.split("_")[0] + '_Frame.csv')
    kam = readKam(resultDir + subject + "\Trial_"+trialNum+"_"+foot+"_1.txt", rate)
    # for iotpos in iotCols:
    #     imupos = iot2imuPos[iotCols.index(iotpos)]
    #     getIotImuLagFFT(iotdata[iotpos], imudata[imupos], flags[iotCols.index(iotpos)])


    trialRange = frameRange.loc[(frameRange["Trial No."] == int(trialNum)) & (frameRange["L/R_"+foot] == 1)]
    LOn = trialRange["On"].iat[0]
    LOff = trialRange["Off"].iat[0]
    if (rate == 100):
        usableLen = LOff - LOn
    else:
        usableLen = 2 * (LOff - LOn)
        LOn = 2 * LOn
        LOff = LOn + usableLen
    '''
    imuSyncOnIndex = imudata[imudata["syncOn"] == 1].index.tolist()
    imuSyncLag = imuSyncOnIndex[0] - 1

    imuOn = imuSyncLag + LOn
    imuOff = imuSyncLag + LOn + usableLen
    pos1 = "LLACy"
    pos2 = "RMACy"
    imuPos1 = iot2imuPos[iotCols.index(pos1)]
    imuPos2 = iot2imuPos[iotCols.index(pos2)]
    maxtab, mintab = peakdet(kam[LOn:LOff].y, 0.002)

    pl.figure(figsize=(19.20, 9.06))
    p1 = pl.subplot(311)
    p2 = pl.subplot(312)
    p3 = pl.subplot(313)

    p1.set_title("LLAC-y & RMAC-y")
    p2.set_title("KAM")
    p3.set_title("LLAC-y + RMAC-y")

    p1.plot(imudata[imuPos1][imuOn:imuOff], color='red')
    p1.plot(imudata[imuPos2][imuOn:imuOff], color='blue')
    p2.plot(kam[LOn:LOff].y)
    p2.scatter(array(maxtab)[:, 0] + 3 + LOn, array(maxtab)[:, 1], color='blue')
    imuCom = imudata[imuPos1][imuOn:imuOff] + imudata[imuPos2][imuOn:imuOff]#
    p3.plot(imuCom)

    r = np.corrcoef(imuCom, -kam[LOn:LOff].y)[0,1]
    print(r)

    pl.show()
'''
    maxtab, mintab = peakdet(kam.y,0.002)


    imuSyncOnIndex = imudata[imudata["syncOn"] == 1].index.tolist()
    imuSyncLag = imuSyncOnIndex[0] - 1

    imuOn = imuSyncLag + LOn
    imuOff = imuSyncLag + LOn + usableLen


    lags = peakLag(imudata, iotdata)
    print("lags:",lags)
    # iot2kam = getIot2Kam(lags, iotdata, kam, imuOn, imuOff, 50)
    # iot2kam.to_csv("train/" + subject + "_trial" + trialNum + "_" + foot + ".txt",sep="\t")
    # llacx = iot2kam.LLACx
    # pl.plot(llacx)
    # pl.show()

    pos1 = "LLGYz"
    pos2 = "RLGYz"
    imuPos1 = iot2imuPos[iotCols.index(pos1)]
    imuPos2 = iot2imuPos[iotCols.index(pos2)]
    flag1 = flags[iotCols.index(pos1)]
    flag2 = flags[iotCols.index(pos2)]

    iotOnLL = imuOn - lags[pos1] - 10
    iotOffLL = imuOff - lags[pos1] + 10
    iotOnLM = imuOn - lags[pos2] - 10
    iotOffLM = imuOff - lags[pos2] + 10

    print("LOn,LOff:",LOn,LOff)
    print("imuOn,imuOff:",imuOn,imuOff)
    print("iotOnLL,iotOffLL,iotOnLM,iotOffLM:",iotOnLL,iotOffLL,iotOnLM,iotOffLM)
    print("usableLen",usableLen)
    print("imuSyncLag",imuSyncLag)

    # getIotImuLag(iotdata[pos1][iotOnLL:iotOffLL], imudata[imuPos1][imuOn:imuOff], flags[iotCols.index(pos1)])
    # getIotImuLagFFT(iotdata[pos2][iotOnLL:iotOffLL], imudata[imuPos2][imuOn:imuOff], flags[iotCols.index(pos2)])
    pl.figure(figsize = (19.20,9.06))
    p7 = pl.subplot(311)
    p8 = pl.subplot(312)
    p9 = pl.subplot(313)

    imumaxindex = array(maxtab)[:, 0]+imuSyncLag-3
    maxvalue1 = list(map(lambda x: imudata[imuPos1][x],imumaxindex))
    maxvalue2 = list(map(lambda x: imudata[imuPos2][x], imumaxindex))
    # maxvalue1 = list(map(lambda x: flag1*iotdata[pos1][x], imumaxindex+lags[pos1]))
    # maxvalue2 = list(map(lambda x: flag2*iotdata[pos2][x], imumaxindex+lags[pos2]))
    imuindex = imudata.index.tolist()
    imuindex = range(len(imuindex))

    imumax, imumin = peakdet(imudata[imuPos1], 500)

    p7.plot(imudata[imuPos1], color="red")  # [imuOn:imuOff])
    # p7.plot(flag1*iotdata[pos1], color="red")
    # p7.scatter(imumax[:,0], imumax[:,1], color='green')
    # p7.scatter(imumaxindex, maxvalue1, color='red')
    # p7.plot(flag2*iotdata[pos2], color="blue")
    p7.plot(imudata[imuPos2], color='blue')  # [imuOn:imuOff])
    p7.scatter(imumaxindex, maxvalue2, color='blue')
    # p6.plot(iotdata[pos1].index + lags[pos1], flag1*iotdata[pos1], color="red")
    # p7.plot(iotdata[pos2].index + lags[pos2], flag2*iotdata[pos2], color="red")
    p8.plot(imudata[imuPos2] + imudata[imuPos1])

    p7.plot(kam.index+imuSyncLag-3,kam.y*20000+800, color='black')
    # p7.plot(kam.index+imuSyncLag-3,kam.y*10000)
    p8.plot(kam.index+imuSyncLag-3,kam.y*20000+2000)
    p8.scatter(array(maxtab)[:, 0]+imuSyncLag, array(maxtab)[:, 1]*20000+2000, color='blue')
    peakkam = list(filter(lambda x:x>0.01,maxtab[:,1]))
    p9.plot(kam.y)
    print(peakkam)

    pl.show()
    # pl.savefig(subject + "-trial" + trialNum + ".png")

