import json
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import matplotlib.pyplot as plt

modelL = joblib.load('model/' + 'RandomForest-iot-allData-L.model')
modelR = joblib.load('model/' + 'RandomForest-iot-allData-R.model')

addr2pos = {"90BDB8B2-4737-1EB3-8AC7-756943596524": "LL", "7DFEC5A2-697F-482F-C6A8-9A0450ECC674": "LM", \
            "9DBD3CB0-12E9-D9F8-823A-EEAEA7A840D1": "RL", "8D6C8805-B13E-9F56-5B88-1DC63407869F": "RM", \
            "ADF9BB24-6649-8CC1-AF67-8AACB4F146EC": "RM", "92F37785-2E73-2E79-0F6F-856BDEC44D29":"RL"}

dataCols = dict(LLACx=[], LLACy=[], LLACz=[], LLGYx=[], LLGYy=[], LLGYz=[], LMACx=[], LMACy=[], LMACz=[],
                LMGYx=[], LMGYy=[], LMGYz=[], RLACx=[], RLACy=[], RLACz=[], RLGYx=[], RLGYy=[], RLGYz=[],
                RMACx=[], RMACy=[], RMACz=[], RMGYx=[], RMGYy=[], RMGYz=[])

subjectInfoCol = ['age', 'mass', 'height', 'Lleglen', 'LkneeWid', 'LankleWid', 'Rleglen', 'RkneeWid', 'RankleWid',
               'gender_F', 'gender_M']
subjectInfo = [56,66.3,1631,815,102,87,820,102,84,0,1]
countFlag = 0

def getPreData(dataCols):
    values = dataCols.values()
    colNum = min(list(map(lambda x: len(x), values)))
    msgCols = dict(LLACx=[], LLACy=[], LLACz=[], LLGYx=[], LLGYy=[], LLGYz=[], LMACx=[], LMACy=[], LMACz=[],
                LMGYx=[], LMGYy=[], LMGYz=[], RLACx=[], RLACy=[], RLACz=[], RLGYx=[], RLGYy=[], RLGYz=[],
                RMACx=[], RMACy=[], RMACz=[], RMGYx=[], RMGYy=[], RMGYz=[])
    for col in dataCols:
        msgCols[col] = dataCols[col][0:colNum]
        dataCols[col] = dataCols[col][colNum:]
    pdMsgCols =pd.DataFrame(msgCols)
    pdSubInfo = pd.DataFrame([subjectInfo] * colNum, columns = subjectInfoCol)
    # pdSubInfo.columns = subjectInfoCol
    pdMsgCols = pd.concat([pdMsgCols,pdSubInfo], axis=1)
    return pdMsgCols

def msgProcess(msg):
    global colFlag
    tempData = json.loads(msg)
    index = 1000 if (tempData['sensor'] == 'accelerometer') else 1
    x = float(tempData['value']['x']) * index
    y = float(tempData['value']['y']) * index
    z = float(tempData['value']['z']) * index
    pos = addr2pos[tempData['address']]

    sensor = 'AC' if (index == 1000) else 'GY'
    dataCols[pos + sensor + 'x'].append(x)
    dataCols[pos + sensor + 'y'].append(y)
    dataCols[pos + sensor + 'z'].append(z)

if __name__ == '__main__':
    rawData = open('test.txt', 'r')  # raw data file
    rawDataFile = rawData.readlines()
    # Lkamall = pd.DataFrame([],columns=['kamL'])
    # Rkamall = pd.DataFrame([],columns=['kamR'])
    Lkamall = np.array([])
    Rkamall = np.array([])

    p1 = plt.subplot(211)
    p2 = plt.subplot(212)

    for line in rawDataFile:
        msgProcess(line)
        countFlag = countFlag + 1

        if(countFlag == 100):
            countFlag = 0
            preData = getPreData(dataCols)
            Lkam = modelL.predict(preData)
            Rkam = modelR.predict(preData)

            Lkamall = np.concatenate((Lkamall, Lkam))
            Rkamall = np.concatenate((Rkamall, Rkam))

            p1.plot(Lkamall)
            p2.plot(Rkamall)
            plt.draw()
            # plt.show()

    if(countFlag > 0):
        countFlag = 0
        preData = getPreData(dataCols)
        Lkam = modelL.predict(preData)
        Rkam = modelR.predict(preData)

        Lkamall =  np.concatenate((Lkamall, Lkam))
        Rkamall = np.concatenate((Rkamall, Rkam))

        p1.plot(Lkamall)
        p2.plot(Rkamall)
        plt.draw()
        plt.show()










