from dataProcessing import *
from sklearn.externals import joblib
import os

modelL = joblib.load('model/' + 'RandomForest-iot-allData-L-LM.model')
modelR = joblib.load('model/' + 'RandomForest-iot-allData-R-RL.model')

if __name__ == '__main__':
    iotdatadir = "iot/"

    iotSubjects = ['iot-S10', 'iot-S11', 'iot-S16', 'iot-S18', 'iot-S21', 'iot-S22', 'iot-S23', 'iot-S33', 'iot-S35',
                   'iot-S7', 'iot-S8', 'iot-S12']
    subjectFile = 'S10_0306-subject5'
    subjectName = 'S10_Fu'
    trialNum = 5
    iotFiles = os.listdir(iotdatadir + subjectFile + '/')
    tempIotcols = ['LLACx', 'LLACy', 'LLACz', 'LLGYx', 'LLGYy', 'LLGYz'
    , 'LMACx', 'LMACy', 'LMACz', 'LMGYx', 'LMGYy', 'LMGYz'
    , 'RLACx', 'RLACy', 'RLACz', 'RLGYx', 'RLGYy', 'RLGYz'
    , 'RMACx', 'RMACy', 'RMACz', 'RMGYx', 'RMGYy', 'RMGYz','KAM','mass']

    infoCols = ['age', 'mass', 'height', 'Lleglen', 'LkneeWid', 'Rleglen', 'LankleWid', 'RkneeWid',
                'RankleWid', 'gender_F', 'gender_M']

    filename = 'subjectInfo.txt'
    info = pd.read_table(filename)
    info = pd.get_dummies(info)

    subjectNum = subjectFile.split("_")[0]
    iotSubjectFile = iotFiles[trialNum - 1]  # iot file name
    subjectInfo = info.loc[info['subject'] == int(subjectNum[1:])]

    rate=100
    staticData = readIotData(iotdatadir + subjectFile + "/modified-static.txt", rate)
    caliData, rotMat = getCaliData(staticData,iotstd)

    iotdata = readIotData(iotdatadir + subjectFile + '/' + iotSubjectFile, rate)
    acCaliIotdata = calibrateData(iotdata, rotMat)
    caliIotdata = gyroCali(caliData, acCaliIotdata)

    usableLen = (caliIotdata.shape)[0]
    value = subjectInfo.iloc[0, 1:].values
    test = [list(value)] * usableLen
    test = pd.DataFrame(test)
    test.columns = infoCols

    # trialNumPd = [trialNum] * usableLen
    # trialNumPd = pd.DataFrame(trialNumPd)
    # trialNumPd.columns = ['trialNum']
    #
    # subnum = pd.DataFrame([subjectNum] * usableLen)
    # subnum.columns = ['subject']

    data = pd.concat([caliIotdata, test], axis=1)
    # data.to_csv(iotdatadir + subjectFile + '/' + subjectFile + '-' + str(trialNum) + ".txt", sep="\t", float_format='%.6f', index=None)
    axisL = 'LM'
    testListL = list(filter(lambda x: axisL in x, tempIotcols))
    testListL.extend(infoCols)
    axisR = 'RL'
    testListR = list(filter(lambda x: axisR in x, tempIotcols))
    testListR.extend(infoCols)
    Lkam = modelL.predict(data[testListL])
    Rkam = modelR.predict(data[testListR])

    frameFile = subjectNum + "_Frame.csv"  # S12_Frame.csv
    frameRange = getFrame('test/' + subjectName + "/" + frameFile)  # result/S12_Lau/S12_Frame.csv

    fileNameL = 'Trial_'+str(trialNum) + '_L_1.txt'
    kamdataL = readKam('test/' + subjectName + "/" + fileNameL, rate)
    fileNameR = 'Trial_' + str(trialNum) + '_R_1.txt'
    kamdataR = readKam('test/' + subjectName + "/" + fileNameR, rate)

    # trialRangeL = frameRange.loc[(frameRange["Trial No."] == int(trialNum-1))
    #                             & (frameRange["L/R_" + 'L'] == 1)]
    # trialRangeR = frameRange.loc[(frameRange["Trial No."] == int(trialNum-1))
    #                              & (frameRange["L/R_" + 'R'] == 1)]


    p1 = pl.subplot(221)
    p2 = pl.subplot(223)
    p3 = pl.subplot(222)
    p4 = pl.subplot(224)
    p1.plot(Lkam, 'r')
    p1.plot(0.0001 * data.RMGYz)
    p2.plot(Rkam, 'g')
    p2.plot(0.0001*data.LLGYz)
    p3.plot(kamdataL.y)
    p4.plot(kamdataR.y)
    pl.show()