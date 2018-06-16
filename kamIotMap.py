from dataProcessing import *

if __name__ == '__main__':
    iotdatadir = "iot/"
    norxDir = "noraxon/"
    resultDir = "test/"
    testdir = "temptest/"
    subject = "S10_Fu"
    dir = "S10_0306-subject5"
    trialNum = '6'
    foot = "L"

    tempIotcols = ['LLACx', 'LLACy', 'LLACz', 'LLGYx', 'LLGYy', 'LLGYz'
        , 'LMACx', 'LMACy', 'LMACz', 'LMGYx', 'LMGYy', 'LMGYz'
        , 'RLACx', 'RLACy', 'RLACz', 'RLGYx', 'RLGYy', 'RLGYz'
        , 'RMACx', 'RMACy', 'RMACz', 'RMGYx', 'RMGYy', 'RMGYz', 'KAM', 'mass']
    infoCols = ['subject', 'age', 'mass', 'height', 'Lleglen', 'LkneeWid', 'Rleglen', 'LankleWid', 'RkneeWid',
                   'RankleWid', 'gender_F', 'gender_M']

    filename = 'subjectInfo.txt'
    info = pd.read_table(filename)
    info = pd.get_dummies(info)

    subjects = os.listdir(testdir[:-1])  # kam file name is same as noraxon file name
    allData = None
    for subjectName in subjects:  # for every subject, subject name : S12_Lau
        subjectData = None
        print(subjectName)
        subjectNum = subjectName.split("_")[0]  # subject number : S12

        subjectInfo = info.loc[info['subject'] == int(subjectNum[1:])]

        imuTrialList = os.listdir(norxDir + subjectName)  # imu data of trials
        resultList = filterKamList(resultDir + subjectName)  # kam data of trials
        staticData = readStaticData(norxDir + subjectName + "/static.txt")
        caliData, rotMat = getCaliData(staticData)
        # caliData = getCaliData(staticData)
        # caliData = caliData.iloc[:, 2:]
        print(resultList)

        frameFile = subjectNum + "_Frame.csv"  # S12_Frame.csv
        frameRange = getFrame(resultDir + subjectName + "/" + frameFile)  # result/S12_Lau/S12_Frame.csv

        for kamFile in resultList:
            fileName = kamFile["file"]
            trialNum = kamFile["trialNum"]
            foot = kamFile["foot"]
            tail = kamFile["tail"]
            print(fileName)

            trialRange = frameRange.loc[(frameRange["Trial No."] == int(trialNum))
                                        & (frameRange["L/R_" + foot] == 1)]
            # get imu data, iot data and imu sampling rate
            rate, imudata = readImuData(norxDir + subjectName + "/Trial_" + trialNum + ".txt")

            kamdata = readKam(resultDir + subjectName + "/" + fileName,imurate = rate)
            LOn = trialRange["On"].iat[0]
            LOff = trialRange["Off"].iat[0]
            if (rate == 100):
                usableLen = LOff - LOn
            else:
                usableLen = 2 * (LOff - LOn)
                LOn = 2 * LOn
                LOff = LOn + usableLen

            shift = int(usableLen / 6)
            usableLen = usableLen - 2 * shift
            LOn = LOn + shift
            LOff = LOn + usableLen

            imuSyncOnIndex = imudata[imudata["syncOn"] == 1].index.tolist()
            imuSyncLag = imuSyncOnIndex[0] - 1


            imuOn = imuSyncLag + LOn +shift
            imuOff = imuOn + usableLen
#======
            usableImudata = imudata[imuOn:imuOff].reset_index().iloc[:, 3:]
            usableImudata = calibrateData(usableImudata, rotMat)
            usableImudata = gyroCali(caliData,usableImudata)
            # usableImudata = usableImudata.sub(caliData.iloc[0, :])
            kamy = kamdata[LOn:LOff].y
            kamy = kamy.reset_index().iloc[:, 1]
            # mass = pd.DataFrame([subjectMass[subjectNum]] * usableLen)
            # mass.columns = ['mass']

            #subject info data
            value = subjectInfo.iloc[0, 1:10].values
            test = [list(value)] * usableLen
            test = pd.DataFrame(test)
            test.columns = infoCols[1:10]

            subnum = pd.DataFrame([subjectNum] * usableLen)
            subnum.columns = ['subject']

            data = pd.concat([subnum,usableImudata,test,kamy], axis=1)

            # pos1 = "LLGYz"
            # pos2 = "RMGYz"
            # imuPos1 = iot2imuPos[iotCols.index(pos1)]
            # imuPos2 = iot2imuPos[iotCols.index(pos2)]
            # # imuCom = imudata[imuPos1][imuOn:imuOff] + imudata[imuPos2][imuOn:imuOff]
            # # imuCom = imuCom.reset_index().iloc[:, 1]
            # pos1imu = imudata[imuPos1][imuOn:imuOff].reset_index().iloc[:, 1]
            # pos2imu = imudata[imuPos2][imuOn:imuOff].reset_index().iloc[:, 1]
            # kamy = kamdata[LOn:LOff].y
            # kamy = kamy.reset_index().iloc[:, 1]
            # mass = [subjectMass[subjectNum]] * (usableLen)
            # subMas = pd.Series(mass,name='mass')
            # data = pd.concat([pos1imu,pos2imu,kamy,subMas], axis=1)
            # data.columns = ['LLGYz','RMGYz','KAMy','mass']

            subjectData = pd.concat([subjectData,data])
            allData = pd.concat([allData,data])

        subjectData.to_csv("kam2cali/" + subjectNum + ".txt", sep="\t",float_format='%.6f',index=None)#, header=None, index=None)
    # allData.to_csv("kam2cali/" + "caliAll.txt", sep="\t", float_format='%.6f',
    #                    index=None)  # , header=None, index=None)