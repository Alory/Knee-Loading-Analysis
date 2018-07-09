from dataProcessing import *
from sklearn.externals import joblib
import os

modelL = joblib.load('model/' + 'RandomForest-70%-allData-L.model')
modelR = joblib.load('model/' + 'RandomForest-70%-allData-R.model')

tempIotcols = ['LLACx', 'LLACy', 'LLACz', 'LLGYx', 'LLGYy', 'LLGYz'
        , 'LMACx', 'LMACy', 'LMACz', 'LMGYx', 'LMGYy', 'LMGYz'
        , 'RLACx', 'RLACy', 'RLACz', 'RLGYx', 'RLGYy', 'RLGYz'
        , 'RMACx', 'RMACy', 'RMACz', 'RMGYx', 'RMGYy', 'RMGYz','mass']
infoCols = ['subject', 'age', 'mass', 'height', 'Lleglen', 'LkneeWid', 'Rleglen', 'LankleWid', 'RkneeWid',
                'RankleWid', 'gender_F', 'gender_M']
infoFile = 'subjectInfo.txt'
out = 'noraxon/'

if __name__ == '__main__':
    testList = tempIotcols[0:24]
    names = ['S7', 'S8', 'S10', 'S11', 'S16', 'S21', 'S22', 'S23', 'S26', 'S28', 'S29', 'S30', 'S31', 'S32', 'S33',
             'S35', 'S37', 'S38', 'S39', 'S40', 'S41']
    subjects = os.listdir(out[:-1])
    subname = 'S16_Lam'
    rate = 100
    trialNum = 6

    staticData = readStaticData(out + subname + "/static.txt")
    caliData, rotMat = getCaliData(staticData, std)

    imurate,rawdata = readImuData(out + subname + '/Trial_' + str(trialNum) + '.txt')
    rawdata = rawdata[testList]
    data = calibrateData(rawdata, rotMat)
    caliData = gyroCali(caliData, data)

    info = pd.read_table(infoFile)
    info = pd.get_dummies(info)

    subjectNum = subname.split("_")[0]
    subjectInfo = info.loc[info['subject'] == int(subjectNum[1:])]
    value = subjectInfo.iloc[0, 1:].values
    usableLen = (caliData.shape)[0]
    test = [list(value)] * usableLen
    test = pd.DataFrame(test)
    test.columns = infoCols[1:]

    data = pd.concat([caliData,test], axis=1)
    # data.to_csv('111.txt',sep='\t')

    Lkam = modelL.predict(data)
    Rkam = modelR.predict(data)

    frameFile = subjectNum + "_Frame.csv"  # S12_Frame.csv
    frameRange = getFrame('test/' + subname + "/" + frameFile)  # result/S12_Lau/S12_Frame.csv

    fileNameL = 'Trial_' + str(trialNum) + '_L_1.txt'
    kamdataL = readKam('test/' + subname + "/" + fileNameL, rate)
    fileNameR = 'Trial_' + str(trialNum) + '_R_1.txt'
    kamdataR = readKam('test/' + subname + "/" + fileNameR, rate)

    p1 = pl.subplot(221)
    p2 = pl.subplot(223)
    p3 = pl.subplot(222)
    p4 = pl.subplot(224)
    p1.plot(Lkam, 'r')
    p1.plot(0.00005 * data.RMGYz)
    p2.plot(Rkam, 'g')
    p2.plot(0.00005 * data.LLGYz)
    p3.plot(kamdataL.y)
    p4.plot(kamdataR.y)
    pl.show()