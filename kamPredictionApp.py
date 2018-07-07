import json
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from scipy import interpolate
from math import acos, atan2, cos, pi, sin

hongkongG = 978.5
std = np.array([hongkongG, 0, 0])

iotCols = ['LLACx', 'LLACy', 'LLACz', 'LLGYx', 'LLGYy', 'LLGYz'
    , 'LMACx', 'LMACy', 'LMACz', 'LMGYx', 'LMGYy', 'LMGYz'
    , 'RLACx', 'RLACy', 'RLACz', 'RLGYx', 'RLGYy', 'RLGYz'
    , 'RMACx', 'RMACy', 'RMACz', 'RMGYx', 'RMGYy', 'RMGYz']

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

def getCaliData(staticData):
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
        R = rotation_matrix(data, std)
        index = np.linalg.norm(std) / np.linalg.norm(data)
        rotMat[axis] = [R,index]


    return caliData,rotMat

# input trial data and the rotate matrix dictionary, output the calibrated data
# this function is to use rotation matrix for acclerometer calibration
def calibrateData(data,rotMat):
    acAxis = {'llac': ['LLACx', 'LLACy', 'LLACz'],
              'lmac': ['LMACx', 'LMACy', 'LMACz'],
              'rlac': ['RLACx', 'RLACy', 'RLACz'],
              'rmac': ['RMACx', 'RMACy', 'RMACz']}

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
    acCols = list(filter(lambda x: 'AC' in x, iotCols))

    cali = zeroOffset.iloc[0, :]
    cali[acCols] = 0.0
    data = data.sub(cali)

    return data


def getPreData(dataCols,caliData,rotMat):
    values = dataCols.values()
    colNum = min(list(map(lambda x: len(x), values)))
    msgCols = dict(LLACx=[], LLACy=[], LLACz=[], LLGYx=[], LLGYy=[], LLGYz=[], LMACx=[], LMACy=[], LMACz=[],
                LMGYx=[], LMGYy=[], LMGYz=[], RLACx=[], RLACy=[], RLACz=[], RLGYx=[], RLGYy=[], RLGYz=[],
                RMACx=[], RMACy=[], RMACz=[], RMGYx=[], RMGYy=[], RMGYz=[])
    for col in dataCols:
        msgCols[col] = dataCols[col][0:colNum]
        dataCols[col] = dataCols[col][colNum:]
    pdMsgCols =pd.DataFrame(msgCols)

    pdMsgCols = calibrateData(pdMsgCols, rotMat)
    pdMsgCols = gyroCali(caliData, pdMsgCols)

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
    staticData = iotStaticData('static.txt')
    caliData, rotMat = getCaliData(staticData)

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
            preData = getPreData(dataCols,caliData,rotMat)
            Lkam = modelL.predict(preData)
            Rkam = modelR.predict(preData)

            Lkamall = np.concatenate((Lkamall, Lkam))
            Rkamall = np.concatenate((Rkamall, Rkam))

            p1.plot(Lkamall,'r')
            p2.plot(Rkamall,'g')
            plt.draw()
            # plt.show()

    if(countFlag > 0):
        countFlag = 0
        preData = getPreData(dataCols,caliData,rotMat)
        if((preData.shape)[0] > 0):
            Lkam = modelL.predict(preData)
            Rkam = modelR.predict(preData)

            Lkamall =  np.concatenate((Lkamall, Lkam))
            Rkamall = np.concatenate((Rkamall, Rkam))

            p1.plot(Lkamall, 'r')
            p2.plot(Rkamall, 'g')
            plt.draw()
    plt.show()










