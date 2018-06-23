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
    resultDir = "result/"
    testdir = "test/"
    subject = "S28_Chan"
    trialNum = '13'
    foot = "L"
    subjectNum = int(subject.split("_")[0][1:])

    rate, imudata = readImuData(norxDir + subject + '\Trial_'+trialNum+'.txt')
    # resultList = filterKamList(resultDir + subject)

    # iotdata = readIotData(iotdatadir + dir + "\\" + iotFileList[int(trialNum)-1], rate)
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
        if(subjectNum > 25):
            usableLen = int(usableLen / 2)
            kam = kam.loc[LOn:LOff,:]
            kam = interpolateDfData(kam, usableLen)
            usableLen = (kam.shape)[0]
            LOn = int(LOn/2)
            LOff = LOn + usableLen
    else:
        usableLen = 2 * (LOff - LOn)
        LOn = 2 * LOn
        LOff = LOn + usableLen

    maxtab, mintab = peakdet(kam.y,0.002)


    imuSyncOnIndex = imudata[imudata["syncOn"] == 1].index.tolist()
    imuSyncLag = imuSyncOnIndex[0] - 1
    # imuSyncLag = 0

    imuOn = imuSyncLag + LOn
    imuOff = imuSyncLag + LOn + usableLen


    # lags = peakLag(imudata, iotdata)
    # print("lags:",lags)
    # iot2kam = getIot2Kam(lags, iotdata, kam, imuOn, imuOff, 50)
    # iot2kam.to_csv("train/" + subject + "_trial" + trialNum + "_" + foot + ".txt",sep="\t")
    # llacx = iot2kam.LLACx
    # pl.plot(llacx)
    # pl.show()

    pos1 = "RLGYz"
    pos2 = "LLGYz"
    imuPos1 = iot2imuPos[iotCols.index(pos1)]
    imuPos2 = iot2imuPos[iotCols.index(pos2)]
    flag1 = flags[iotCols.index(pos1)]
    flag2 = flags[iotCols.index(pos2)]

    # iotOnLL = imuOn - lags[pos1] - 10
    # iotOffLL = imuOff - lags[pos1] + 10
    # iotOnLM = imuOn - lags[pos2] - 10
    # iotOffLM = imuOff - lags[pos2] + 10

    print("LOn,LOff:",LOn,LOff)
    print("imuOn,imuOff:",imuOn,imuOff)
    # print("iotOnLL,iotOffLL,iotOnLM,iotOffLM:",iotOnLL,iotOffLL,iotOnLM,iotOffLM)
    print("usableLen",usableLen)
    print("imuSyncLag",imuSyncLag)

    # getIotImuLag(iotdata[pos1][iotOnLL:iotOffLL], imudata[imuPos1][imuOn:imuOff], flags[iotCols.index(pos1)])
    # getIotImuLagFFT(iotdata[pos2][iotOnLL:iotOffLL], imudata[imuPos2][imuOn:imuOff], flags[iotCols.index(pos2)])
    pl.figure(figsize = (19.20,9.06))
    p1 = pl.subplot(321)
    p2 = pl.subplot(323)

    p5 = pl.subplot(325)
    p6 = pl.subplot(322)
    p7 = pl.subplot(324)
    p8 = pl.subplot(326)

    p1.set_title('noraxon ' + imuPos1)
    p2.set_title('noraxon ' + imuPos2)
    p5.set_title('KAM')
    p6.set_title('sync')
    p8.set_title('KAM')

    p1.plot(imudata[imuPos1][imuOn:imuOff])
    p2.plot(imudata[imuPos2][imuOn:imuOff])
    p5.plot(kam[LOn:LOff].y)

    # imumaxindex = array(maxtab)[:, 0]+imuSyncLag
    # maxvalue1 = list(map(lambda x: imudata[imuPos1][x], imumaxindex))
    # maxvalue2 = list(map(lambda x: imudata[imuPos2][x], imumaxindex))
    p6.plot(imudata[imuPos1])  # [imuOn:imuOff])
    # p6.scatter(imumaxindex, maxvalue1, color='blue')
    p7.plot(imudata[imuPos2])  # [imuOn:imuOff])
    # p7.scatter(imumaxindex, maxvalue2, color='blue')

    p8.plot(kam.y)
    # p8.scatter(array(maxtab)[:, 0]+3, array(maxtab)[:, 1], color='blue')

    pl.show()
    # pl.savefig(subject + "-trial" + trialNum + ".png")
