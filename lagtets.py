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
    subject = "S22_Chen"
    dir = "S22_0327-subject1"
    trialNum = '2'
    foot = "R"

    rate, imudata = readImuData(norxDir + subject + '\Trial_'+trialNum+'.txt')
    iotFileList = os.listdir(iotdatadir + dir)
    # resultList = filterKamList(resultDir + subject)

    iotdata = readIotData(iotdatadir + dir + "\\" + iotFileList[int(trialNum)-1], rate)
    frameRange = getFrame(resultDir + subject + "\\" + subject.split("_")[0] + '_Frame.csv')
    kam = readKam(resultDir + subject + "\Trial_"+trialNum+"_"+foot+"_12.txt", rate)
    # for iotpos in iotCols:
    #     imupos = iot2imuPos[iotCols.index(iotpos)]
    #     getIotImuLagFFT(iotdata[iotpos], imudata[imupos], flags[iotCols.index(iotpos)])


    trialRange = frameRange.loc[(frameRange["Trial No."] == int(trialNum)) & (frameRange["L/R_"+foot] == 1)]
    LOn = trialRange["On"].iat[0]
    LOff = trialRange["Off"].iat[0]
    if(rate == 100):
        usableLen = LOff - LOn + 1
    else:
        usableLen = 2*(LOff - LOn + 1)
        LOn = 2*LOn - 2
        LOff = LOn + usableLen

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

    pos1 = "RLGYz"
    pos2 = "LLGYz"
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
    p1 = pl.subplot(521)
    p2 = pl.subplot(523)
    p3 = pl.subplot(525)
    p4 = pl.subplot(527)
    p5 = pl.subplot(529)
    p6 = pl.subplot(522)
    p7 = pl.subplot(524)
    p8 = pl.subplot(526)
    p9 = pl.subplot(528)

    p1.set_title('noraxon ' + imuPos1)
    p2.set_title('noraxon ' + imuPos2)
    p3.set_title('iot ' + pos1)
    p4.set_title('iot ' + pos2)
    p5.set_title('KAM')
    p6.set_title('sync')
    p8.set_title('KAM')

    p1.plot(imudata[imuPos1][imuOn:imuOff])
    p2.plot(imudata[imuPos2][imuOn:imuOff])
    p3.plot(flag1*iotdata[pos1][iotOnLL:iotOffLL])#flag1*
    p4.plot(flag2*iotdata[pos2][iotOnLM:iotOffLM])#flag2*
    p5.plot(kam[LOn:LOff].y)

    imumaxindex = array(maxtab)[:, 0]+imuSyncLag
    maxvalue1 = list(map(lambda x:imudata[imuPos1][x],imumaxindex))
    maxvalue2 = list(map(lambda x: imudata[imuPos2][x], imumaxindex))
    p6.plot(imudata[imuPos1])  # [imuOn:imuOff])
    p6.scatter(imumaxindex, maxvalue1, color='blue')
    p7.plot(imudata[imuPos2])  # [imuOn:imuOff])
    p7.scatter(imumaxindex, maxvalue2, color='blue')
    p6.plot(iotdata[pos1].index + lags[pos1], flag1*iotdata[pos1], color="red")
    p7.plot(iotdata[pos2].index + lags[pos2], flag2*iotdata[pos2], color="red")

    p8.plot(kam.y)
    p8.scatter(array(maxtab)[:, 0]+3, array(maxtab)[:, 1], color='blue')
    p9.plot(np.diff(kam.y))

    pl.show()
    # pl.savefig(subject + "-trial" + trialNum + ".png")
