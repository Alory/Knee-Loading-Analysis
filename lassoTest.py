import pandas as pd
import numpy as np
import pylab as pl
from scipy import interpolate
from dataProcessing import *
import scipy.io
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
    subject = "S23_Chen"
    dir = "S23_0327-subject2"
    trialNum = '14'
    foot = "L"
    shift = 10

    subjects = os.listdir(testdir[:-1])#kam file name is same as noraxon file name
    iotFiles = os.listdir(iotdatadir[:-1])
    tempIotcols = ['LLACx', 'LLACy', 'LLACz', 'LLGYx', 'LLGYy', 'LLGYz'
    , 'LMACx', 'LMACy', 'LMACz', 'LMGYx', 'LMGYy', 'LMGYz'
    , 'RLACx', 'RLACy', 'RLACz', 'RLGYx', 'RLGYy', 'RLGYz'
    , 'RMACx', 'RMACy', 'RMACz', 'RMGYx', 'RMGYy', 'RMGYz','KAM','mass']

    for subjectName in subjects:#for every subject, subject name : S12_Lau
        print(subjectName)
        subjectData = pd.DataFrame([])
        labels = pd.DataFrame([])
        keywords = pd.DataFrame([])
        subjectNum = subjectName.split("_")[0]#subject number : S12
        iotSubjectFile = getFile(subjectNum,iotFiles)#iot file name

        iotTrialList = os.listdir(iotdatadir + iotSubjectFile)#iot data of trials
        imuTrialList = os.listdir(norxDir + subjectName)#imu data of trials
        resultList = filterKamList(resultDir + subjectName)#kam data of trials
        print(resultList)

        frameFile = subjectNum + "_Frame.csv"#S12_Frame.csv
        frameRange = getFrame(resultDir + subjectName + "/" + frameFile)#result/S12_Lau/S12_Frame.csv

        for kamFile in resultList:
            fileName = kamFile["file"]
            trialNum = kamFile["trialNum"]
            foot = kamFile["foot"]
            tail = kamFile["tail"]
            print(fileName)

            trialRange = frameRange.loc[(frameRange["Trial No."] == int(trialNum))
                                        & (frameRange["L/R_"+foot] == 1)]
            # get imu data, iot data and imu sampling rate
            rate, imudata = readImuData(norxDir + subjectName + "/Trial_" + trialNum + ".txt")
            iotdata = readIotData(iotdatadir + iotSubjectFile + "/" + iotTrialList[int(trialNum)-1], rate)
            kamdata = readKam(resultDir + subjectName + "/" + fileName,rate)

            LOn = trialRange["On"].iat[0]
            LOff = trialRange["Off"].iat[0]
            if (rate == 100):
                usableLen = LOff - LOn
            else:
                usableLen = 2 * (LOff - LOn)
                LOn = 2 * LOn
                LOff = LOn + usableLen

            usableKam = kamdata[LOn:LOff]#usable kam
            imuSyncOnIndex = imudata[imudata["syncOn"] == 1].index.tolist()
            imuSyncLag = imuSyncOnIndex[0] - 1

            imuOn = imuSyncLag + LOn
            imuOff = imuSyncLag + LOn + usableLen

            lags = peakLag(imudata, iotdata)
            # iot2kam = getIot2Kam(lags, iotdata, kamdata, imuOn, imuOff, shift)
            # iot2kamMat = iot2kam.loc[:,iotCols]
            mass = subjectMass[subjectNum]
            iot2kam = getImu2Kam(imudata, usableKam, imuOn, imuOff,mass)
            # subjectData = pd.concat([subjectData, iot2kam],axis=1)
            subjectData = iot2kam
            tempKey = pd.DataFrame([subjectNum + "_trial" + trialNum + "_" + foot]*26)
            tempLabel = pd.DataFrame([x+ '_' + subjectNum + "_trial" + trialNum + "_"
                                      + foot for x in tempIotcols])

            # keywords = pd.concat((keywords,tempKey))
            # labels = pd.concat((labels,tempLabel))
            keywords = tempKey
            labels = tempLabel
#=======
            # subjectData = subjectData.fillna(0)
            colValues = list(range(len(subjectData.columns.values)))
            colValues = np.array(colValues)
            timeSeriesData = list(map(lambda x: np.array(subjectData.iloc[:, x]), colValues))
            cols = subjectData.columns.values

            labels = np.array(labels.iloc[:,0])
            labels = labels.reshape(labels.size, 1)

            keywords = np.array(keywords.iloc[:,0])
            keywords = keywords.reshape(keywords.size,1)
            struct = {"timeSeriesData":timeSeriesData,"labels":labels,"keywords":keywords}
            # scipy.io.savemat("train/" + subjectNum + ".mat",struct)
            scipy.io.savemat("train/" + subjectNum + "_trial" + trialNum + "_" + foot + ".mat", struct)
        # iot2kam.to_csv("train/" + subjectNum + "_trial" + trialNum + "_" + foot + ".txt", sep="\t",float_format='%.3f')