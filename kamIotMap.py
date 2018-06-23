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
    shift = 0

    subjects = os.listdir(testdir[:-1])#kam file name is same as noraxon file name
    # subjects = ['S32_Leng']
    iotFiles = os.listdir(iotdatadir[:-1])
    tempIotcols = ['LLACx', 'LLACy', 'LLACz', 'LLGYx', 'LLGYy', 'LLGYz'
    , 'LMACx', 'LMACy', 'LMACz', 'LMGYx', 'LMGYy', 'LMGYz'
    , 'RLACx', 'RLACy', 'RLACz', 'RLGYx', 'RLGYy', 'RLGYz'
    , 'RMACx', 'RMACy', 'RMACz', 'RMGYx', 'RMGYy', 'RMGYz','KAM','mass']

    infoCols = ['subject', 'age', 'mass', 'height', 'Lleglen', 'LkneeWid', 'Rleglen', 'LankleWid', 'RkneeWid',
                'RankleWid', 'gender_F', 'gender_M']

    filename = 'subjectInfo.txt'
    info = pd.read_table(filename)
    info = pd.get_dummies(info)
    allData = None
    allDataL = None
    allDataR = None
    for subjectName in subjects:#for every subject, subject name : S12_Lau
        print(subjectName)
        subjectData = None
        subjectDataL = pd.DataFrame([])
        subjectDataR = pd.DataFrame([])

        subjectNum = subjectName.split("_")[0]#subject number : S12
        iotSubjectFile = getFile(subjectNum,iotFiles)#iot file name
        subjectInfo = info.loc[info['subject'] == int(subjectNum[1:])]
        if(iotSubjectFile == -1):
            continue

        iotTrialList = os.listdir(iotdatadir + iotSubjectFile)#iot data of trials
        imuTrialList = os.listdir(norxDir + subjectName)#imu data of trials
        resultList = filterKamList(resultDir + subjectName)#kam data of trials
        print(resultList)

        if(subjectNum != 'S18'):
            norxRate,norxstaticData = readImuData(norxDir + subjectName + "/static.txt")
            staticData = readIotData(iotdatadir + iotSubjectFile + "/modified-static.txt",norxRate)
            caliData, rotMat = getCaliData(staticData)

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
                if (int(subjectNum[1:]) > 25):
                    kamdata = kamdata.loc[LOn:LOff,:]
                    kamdata = interpolateDfData(kamdata, int(usableLen / 2))
                    usableLen = (kamdata.shape)[0]
                    LOn = int(LOn / 2)
                    LOff = LOn + usableLen
            else:
                usableLen = 2 * (LOff - LOn)
                LOn = 2 * LOn
                LOff = LOn + usableLen

            # shift = int(usableLen / 6)
            # usableLen = usableLen - 2 * shift
            # LOn = LOn + shift
            # LOff = LOn + usableLen

            usableKam = kamdata[LOn:LOff]#usable kam
            imuSyncOnIndex = imudata[imudata["syncOn"] == 1].index.tolist()
            imuSyncLag = imuSyncOnIndex[0] - 1

            imuOn = imuSyncLag + LOn
            imuOff = imuSyncLag + LOn + usableLen

            lags = peakLag(imudata, iotdata)

            # iot2kamMat = iot2kam.loc[:,iotCols]
            kamy = kamdata[LOn:LOff].y
            kamy = kamy.reset_index().iloc[:, 1]
            usableLen = (kamy.shape)[0]
            iot2kam = getIot2Kam(lags,iotdata,imuOn,imuOff)
            if (ifcontainNan(iot2kam)):
                continue

            if (subjectNum != 'S18'):
                iot2kam = calibrateData(iot2kam, rotMat)
                iot2kam = gyroCali(caliData, iot2kam)

            # subject info data
            value = subjectInfo.iloc[0, 1:].values
            test = [list(value)] * usableLen
            test = pd.DataFrame(test)
            test.columns = infoCols[1:]

            trialNumPd = [trialNum] * usableLen
            trialNumPd = pd.DataFrame(trialNumPd)
            trialNumPd.columns =['trialNum']

            subnum = pd.DataFrame([subjectNum] * usableLen)
            subnum.columns = ['subject']

            data = pd.concat([subnum,trialNumPd,iot2kam,test,kamy],axis=1)
            if (foot == 'L'):
                allDataL = pd.concat([allDataL, data])
                subjectDataL = pd.concat([subjectDataL, data])
            if (foot == 'R'):
                allDataR = pd.concat([allDataR, data])
                subjectDataR = pd.concat([subjectDataR, data])

        if ((subjectDataL.shape)[0] > 0):
            subjectDataL.to_csv("kam2allinfo/" + 'iot-' + subjectNum + "-L.txt", sep="\t", float_format='%.6f', index=None)
        if ((subjectDataR.shape)[0] > 0):
            subjectDataR.to_csv("kam2allinfo/" + 'iot-' + subjectNum + "-R.txt", sep="\t", float_format='%.6f', index=None)
    allDataL.to_csv("kam2allinfo/" + 'iot-' + "allData-L.txt", sep="\t", float_format='%.6f', index=None)
    allDataR.to_csv("kam2allinfo/" + 'iot-' + "allData-R.txt", sep="\t", float_format='%.6f', index=None)