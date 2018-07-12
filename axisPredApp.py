import json
import os

import numpy as np
import pandas as pd
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from scipy import interpolate
from math import acos, atan2, cos, pi, sin

hongkongG = 978.5
iotstd = np.array([0,-hongkongG,0])

iotCols = ['LLACx', 'LLACy', 'LLACz', 'LLGYx', 'LLGYy', 'LLGYz'
    , 'LMACx', 'LMACy', 'LMACz', 'LMGYx', 'LMGYy', 'LMGYz'
    , 'RLACx', 'RLACy', 'RLACz', 'RLGYx', 'RLGYy', 'RLGYz'
    , 'RMACx', 'RMACy', 'RMACz', 'RMGYx', 'RMGYy', 'RMGYz']

modelL = joblib.load('model/' + 'RandomForest-iot-allData-L.model')
modelR = joblib.load('model/' + 'RandomForest-iot-allData-R.model')

# modelL = joblib.load('model/' + 'RandomForest-iot-allData-L-LM.model')
# modelR = joblib.load('model/' + 'RandomForest-iot-allData-R-RL.model')

addr2pos = {"90BDB8B2-4737-1EB3-8AC7-756943596524": "LL", "7DFEC5A2-697F-482F-C6A8-9A0450ECC674": "LM", \
            "9DBD3CB0-12E9-D9F8-823A-EEAEA7A840D1": "RL", "8D6C8805-B13E-9F56-5B88-1DC63407869F": "RM", \
            "ADF9BB24-6649-8CC1-AF67-8AACB4F146EC": "RM", "92F37785-2E73-2E79-0F6F-856BDEC44D29":"RL"}

dataCols = dict(LLACx=[], LLACy=[], LLACz=[], LLGYx=[], LLGYy=[], LLGYz=[], LMACx=[], LMACy=[], LMACz=[],
                LMGYx=[], LMGYy=[], LMGYz=[], RLACx=[], RLACy=[], RLACz=[], RLGYx=[], RLGYy=[], RLGYz=[],
                RMACx=[], RMACy=[], RMACz=[], RMGYx=[], RMGYy=[], RMGYz=[])

subjectInfoCol = ['age', 'mass', 'height', 'Lleglen', 'LkneeWid', 'LankleWid', 'Rleglen', 'RkneeWid', 'RankleWid',
               'gender_F', 'gender_M']
subjectInfo = [58,61.5,1546,793,90,93,810,95,77,1,0]


def iotStaticData(filename):
    dataCols = dict(LLACx=[], LLACy=[], LLACz=[], LLGYx=[], LLGYy=[], LLGYz=[], LMACx=[], LMACy=[], LMACz=[],
                    LMGYx=[], LMGYy=[], LMGYz=[], RLACx=[], RLACy=[], RLACz=[], RLGYx=[], RLGYy=[], RLGYz=[], RMACx=[],
                    RMACy=[], RMACz=[], RMGYx=[], RMGYy=[], RMGYz=[])


    rawData = open(filename, 'r')  # raw data file
    rawDataFile = rawData.readlines()

    try:
        for line in rawDataFile:
            tempData = json.loads(line)
            index = 1000 if (tempData['sensor'] == 'accelerometer') else 1
            x = float(tempData['value']['x']) * index
            y = float(tempData['value']['y']) * index
            z = float(tempData['value']['z']) * index
            pos = addr2pos[tempData['address']]

            sensor = 'AC' if (index == 1000) else 'GY'
            dataCols[pos + sensor + 'x'].append(x)
            dataCols[pos + sensor + 'y'].append(y)
            dataCols[pos + sensor + 'z'].append(z)

    finally:
        rawData.close()
        values = dataCols.values()
        colNum = min(list(map(lambda x: len(x), values)))
        for col in dataCols:
            dataCols[col] = dataCols[col][0:colNum]

        staticData = pd.DataFrame(dataCols)
        return staticData



def interpolateData(data):
    m = len(data)
    n = len(data[0])
    x = np.linspace(1, m, m)
    y = np.linspace(1, n, n)
    xnew = np.linspace(0, m, 2 * m)
    f = interpolate.interp2d(y, x, data, kind='linear')
    newData = f(y, xnew)
    return newData

def readIotData(filename, imuRate=100):
    if (imuRate == 100):
        data = pd.read_table(filename)  # "test.txt"
    else:
        rawData = np.genfromtxt(filename, delimiter="\t", skip_header=1)
        npdata = interpolateData(rawData)
        data = pd.DataFrame(npdata)
        data.columns = iotCols
    # for pos in iotCols:
    #     data[pos] = signal.savgol_filter(data[pos], 7, 3)
    return data

def rotation_matrix(vector_orig, vector_fin):
    """Calculate the rotation matrix required to rotate from one vector to another.
    For the rotation of one vector to another, there are an infinit series of rotation matrices
    possible.  Due to axially symmetry, the rotation axis can be any vector lying in the symmetry
    plane between the two vectors.  Hence the axis-angle convention will be used to construct the
    matrix with the rotation axis defined as the cross product of the two vectors.  The rotation
    angle is the arccosine of the dot product of the two unit vectors.
    Given a unit vector parallel to the rotation axis, w = [x, y, z] and the rotation angle a,
    the rotation matrix R is::
              |  1 + (1-cos(a))*(x*x-1)   -z*sin(a)+(1-cos(a))*x*y   y*sin(a)+(1-cos(a))*x*z |
        R  =  |  z*sin(a)+(1-cos(a))*x*y   1 + (1-cos(a))*(y*y-1)   -x*sin(a)+(1-cos(a))*y*z |
              | -y*sin(a)+(1-cos(a))*x*z   x*sin(a)+(1-cos(a))*y*z   1 + (1-cos(a))*(z*z-1)  |
    @param R:           The 3x3 rotation matrix to update.
    @type R:            3x3 numpy array
    @param vector_orig: The unrotated vector defined in the reference frame.
    @type vector_orig:  numpy array, len 3
    @param vector_fin:  The rotated vector defined in the reference frame.
    @type vector_fin:   numpy array, len 3
    """

    # Convert the vectors to unit vectors.
    vector_orig = vector_orig / np.linalg.norm(vector_orig)
    vector_fin = vector_fin / np.linalg.norm(vector_fin)

    # The rotation axis (normalised).
    axis = np.cross(vector_orig, vector_fin)
    axis_len = np.linalg.norm(axis)
    if axis_len != 0.0:
        axis = axis / axis_len

    # Alias the axis coordinates.
    x = axis[0]
    y = axis[1]
    z = axis[2]

    # The rotation angle.
    angle = acos(np.dot(vector_orig, vector_fin))

    # Trig functions (only need to do this maths once!).
    ca = cos(angle)
    sa = sin(angle)

    R = np.zeros((3, 3))
    # Calculate the rotation matrix elements.
    R[0, 0] = 1.0 + (1.0 - ca) * (x ** 2 - 1.0)
    R[0, 1] = -z * sa + (1.0 - ca) * x * y
    R[0, 2] = y * sa + (1.0 - ca) * x * z
    R[1, 0] = z * sa + (1.0 - ca) * x * y
    R[1, 1] = 1.0 + (1.0 - ca) * (y ** 2 - 1.0)
    R[1, 2] = -x * sa + (1.0 - ca) * y * z
    R[2, 0] = -y * sa + (1.0 - ca) * x * z
    R[2, 1] = x * sa + (1.0 - ca) * y * z
    R[2, 2] = 1.0 + (1.0 - ca) * (z ** 2 - 1.0)

    return R

def getCaliData(staticData,stdg):
    staticData = staticData[iotCols]
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
        data = (np.array(data))[0]
        # print(data,std)
        R = rotation_matrix(data, stdg)
        index = np.linalg.norm(stdg) / np.linalg.norm(data)
        rotMat[axis] = [R,index]


    return caliData,rotMat

# input trial data and the rotate matrix dictionary, output the calibrated data
# this function is to use rotation matrix for acclerometer calibration
def calibrateData(data,rotMat,senPos=['LL','LM','RL','RM']):
    acAxis = {'llac': ['LLACx', 'LLACy', 'LLACz'],
              'lmac': ['LMACx', 'LMACy', 'LMACz'],
              'rlac': ['RLACx', 'RLACy', 'RLACz'],
              'rmac': ['RMACx', 'RMACy', 'RMACz']}

    acAxis = {key:value for key,value in acAxis.items() if (key.upper())[0:2] in senPos}

    for indexs in data.index:
        for axis in acAxis:
            pos = acAxis[axis]
            R = rotMat[axis][0]
            coef = rotMat[axis][1]
            m = data.loc[indexs, pos]
            out = coef * np.dot(R,m)
            out.shape = [1, 3]
            out = np.array(out)
            data.loc[indexs, pos] = out[0]

    return data

# this function is to calculate the zero drift for gyroscope calibration
def gyroCali(zeroOffset,data):
    cols = list(data.columns)
    acCols = list(filter(lambda x: 'AC' in x, cols))

    usa0drift = zeroOffset[cols]
    cali = usa0drift.iloc[0, :]
    cali[acCols] = 0.0
    data = data.sub(cali)

    return data


def getPreData(currdata,caliData,rotMat,pos):# pos = ['LL', 'LM', 'RL', 'RM']
    usedataCols = {key:value for key,value in currdata.items() if key[0:2] in pos}
    values = usedataCols.values()
    colNum = min(list(map(lambda x: len(x), values)))
    if(colNum == 0):
        return None
    for col in usedataCols:
        usedataCols[col] = usedataCols[col][0:colNum]
    pdMsgCols =pd.DataFrame(usedataCols)

    pdMsgCols = calibrateData(pdMsgCols, rotMat,pos)
    pdMsgCols = gyroCali(caliData, pdMsgCols)

    pdSubInfo = pd.DataFrame([subjectInfo] * colNum, columns = subjectInfoCol)
    # pdSubInfo.columns = subjectInfoCol
    pdMsgCols = pd.concat([pdMsgCols,pdSubInfo], axis=1)
    return pdMsgCols

allData = {}

def msgProcess(msg):
    tempData = json.loads(msg)
    timestamp = tempData['time']
    index = 1000 if (tempData['sensor'] == 'accelerometer') else 1
    x = float(tempData['value']['x']) * index
    y = float(tempData['value']['y']) * index
    z = float(tempData['value']['z']) * index
    pos = addr2pos[tempData['address']]
    sensor = 'AC' if (index == 1000) else 'GY'

    if(len(allData) == 0):
        allData[timestamp] = dict(LLACx=[], LLACy=[], LLACz=[], LLGYx=[], LLGYy=[], LLGYz=[], LMACx=[], LMACy=[],
                                  LMACz=[],LMGYx=[], LMGYy=[], LMGYz=[], RLACx=[], RLACy=[], RLACz=[], RLGYx=[], RLGYy=[],
                                  RLGYz=[],RMACx=[], RMACy=[], RMACz=[], RMGYx=[], RMGYy=[], RMGYz=[])
        allData[timestamp][pos + sensor + 'x'].append(x)
        allData[timestamp][pos + sensor + 'y'].append(y)
        allData[timestamp][pos + sensor + 'z'].append(z)

        return -1
    elif(timestamp in allData): # still have data incoming in this moment
        allData[timestamp][pos + sensor + 'x'].append(x)
        allData[timestamp][pos + sensor + 'y'].append(y)
        allData[timestamp][pos + sensor + 'z'].append(z)

        return -1
    else: # first data of new moment's coming
        allData[timestamp] = dict(LLACx=[], LLACy=[], LLACz=[], LLGYx=[], LLGYy=[], LLGYz=[], LMACx=[], LMACy=[], LMACz=[],
                LMGYx=[], LMGYy=[], LMGYz=[], RLACx=[], RLACy=[], RLACz=[], RLGYx=[], RLGYy=[], RLGYz=[],
                RMACx=[], RMACy=[], RMACz=[], RMGYx=[], RMGYy=[], RMGYz=[])
        allData[timestamp][pos + sensor + 'x'].append(x)
        allData[timestamp][pos + sensor + 'y'].append(y)
        allData[timestamp][pos + sensor + 'z'].append(z)

        currdata = allData[timestamp-1]
        allData[timestamp - 1] = []

        return currdata

allpos = ['LL', 'LM', 'RL', 'RM']

if __name__ == '__main__':
    staticData = iotStaticData('subject/static.txt')
    caliData, rotMat = getCaliData(staticData,iotstd)

    trials = os.listdir('subject')
    triNum = 2

    rawData = open('subject/' + trials[triNum-1], 'r')  # raw data file
    rawDataFile = rawData.readlines()
    # Lkamall = pd.DataFrame([],columns=['kamL'])
    # Rkamall = pd.DataFrame([],columns=['kamR'])
    Lkamall = np.array([])
    Rkamall = np.array([])

    p1 = plt.subplot(211)
    p2 = plt.subplot(212)

    for line in rawDataFile:
        currdata = msgProcess(line)
        if(currdata != -1):
            LpreData = getPreData(currdata, caliData, rotMat, allpos)
            RpreData = getPreData(currdata, caliData, rotMat, allpos)

            if(LpreData is not None and RpreData is not None):
                Lkam = modelL.predict(LpreData)
                Rkam = modelR.predict(RpreData)

                Lkamall = np.concatenate((Lkamall, Lkam))
                Rkamall = np.concatenate((Rkamall, Rkam))

                p1.plot(Lkamall, 'r')
                p2.plot(Rkamall, 'g')
                plt.draw()

    currTime = (json.loads(line))['time']
    currdata = allData[currTime]
    LpreData = getPreData(currdata, caliData, rotMat, allpos)
    RpreData = getPreData(currdata, caliData, rotMat, allpos)

    if (LpreData is not None and RpreData is not None):
        Lkam = modelL.predict(LpreData)
        Rkam = modelR.predict(RpreData)

        Lkamall = np.concatenate((Lkamall, Lkam))
        Rkamall = np.concatenate((Rkamall, Rkam))

        p1.plot(Lkamall, 'r')
        p2.plot(Rkamall, 'g')
        plt.draw()

    plt.show()










